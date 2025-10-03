from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import sys

# Allow running as a script without package context
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.detection import build_detection_model
from models.recognition import CRNN
from data.tokenizer import BengaliWordOCRTokenizer


class EndToEndOCR:
    def __init__(self, detection_model_path: str | None, recognition_model_path: str, tokenizer_path: str):
        # Detection: use pretrained predictor (path optional, predictor not usually loaded from ckpt)
        self.detector = build_detection_model(pretrained=True)
        # Recognition
        self.tok = BengaliWordOCRTokenizer()
        self.tok.load_vocab(Path(tokenizer_path))
        self.recognizer = CRNN(vocab_size=self.tok.vocab_size(), pretrained_backbone=False)
        if recognition_model_path:
            state = torch.load(recognition_model_path, map_location='cpu')
            self.recognizer.load_state_dict(state)
        self.recognizer.eval()

    def detect_words(self, image: np.ndarray) -> List[List[float]]:
        # returns normalized boxes [x1,y1,x2,y2]
        from doctr.io.image import read_img_as_tensor
        import numpy as np
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tmp = Path("_tmp_det.jpg")
        cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        page = read_img_as_tensor(str(tmp))
        preds = self.detector([page])
        tmp.unlink(missing_ok=True)
        boxes: List[List[float]] = []
        for p in preds.pages[0].blocks:
            for l in p.lines:
                for w in l.words:
                    poly = np.array(w.geometry, dtype=float)
                    xs = poly[:, 0]; ys = poly[:, 1]
                    boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        return boxes

    @torch.no_grad()
    def recognize_word(self, crop_image: np.ndarray) -> str:
        # Prepare crop (H=32)
        rgb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = 32.0 / max(1, h)
        new_w = max(int(round(w * scale)), 1)
        rgb = cv2.resize(rgb, (new_w, 32), interpolation=cv2.INTER_AREA)
        # Ensure min width to avoid collapse through pooling
        min_width = 32
        if rgb.shape[1] < min_width:
            pad_w = min_width - rgb.shape[1]
            pad = np.zeros((32, pad_w, 3), dtype=rgb.dtype)
            rgb = np.concatenate([rgb, pad], axis=1)
        img = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        logits = self.recognizer(img)
        ids = self.recognizer.ctc_decode(logits)[0]
        return self.tok.decode_indices(ids)

    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        img = cv2.imread(image_path)
        if img is None:
            return []
        H, W = img.shape[:2]
        boxes = self.detect_words(img)
        results: List[Dict[str, Any]] = []
        for b in boxes:
            x1 = int(round(b[0] * W)); y1 = int(round(b[1] * H))
            x2 = int(round(b[2] * W)); y2 = int(round(b[3] * H))
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            text = self.recognize_word(crop)
            results.append({'box': b, 'text': text})
        return results

    def visualize(self, image: np.ndarray, results: List[Dict[str, Any]], ground_truth: List[Dict[str, Any]] | None = None) -> np.ndarray:
        out = image.copy()
        H, W = out.shape[:2]
        for r in results:
            x1 = int(round(r['box'][0] * W)); y1 = int(round(r['box'][1] * H))
            x2 = int(round(r['box'][2] * W)); y2 = int(round(r['box'][3] * H))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, r['text'], (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return out


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description='Run end-to-end OCR on a folder of images and save visualizations/results')
    ap.add_argument('--images_dir', type=str, required=True)
    ap.add_argument('--rec_model', type=str, required=True, help='Path to trained recognition model_best.pt')
    ap.add_argument('--vocab', type=str, required=True, help='Path to vocab.json')
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--limit', type=int, default=0, help='Optional limit on number of images')
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = EndToEndOCR(detection_model_path=None, recognition_model_path=args.rec_model, tokenizer_path=args.vocab)

    # Collect images
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    imgs = [p for p in images_dir.rglob('*') if p.suffix.lower() in exts]
    if args.limit and args.limit > 0:
        imgs = imgs[: args.limit]

    all_results: Dict[str, Any] = {}
    for p in imgs:
        res = ocr.process_image(str(p))
        all_results[str(p)] = res
        # Save visualization
        img = cv2.imread(str(p))
        if img is not None:
            vis = ocr.visualize(img, res)
            cv2.imwrite(str(out_dir / f"{p.stem}_vis{p.suffix}"), vis)

    # Save results JSON
    (out_dir / 'results.json').write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved OCR results and visualizations to', out_dir)


if __name__ == '__main__':
    main()
