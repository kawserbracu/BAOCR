import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_normalized_box_xyxy(box: Tuple[int, int, int, int], iw: int, ih: int) -> List[float]:
    x1, y1, x2, y2 = box
    # clip
    x1 = max(0, min(x1, iw - 1))
    y1 = max(0, min(y1, ih - 1))
    x2 = max(x1 + 1, min(x2, iw))
    y2 = max(y1 + 1, min(y2, ih))
    return [x1 / iw, y1 / ih, x2 / iw, y2 / ih]


def apply_preprocess(img: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "original":
        return img
    if mode == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    if mode == "highboost":
        alpha = 1.5
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        mask = cv2.subtract(img, blurred)
        boosted = cv2.add(img, cv2.multiply(mask, np.array([alpha], dtype=np.float32)))
        boosted = np.clip(boosted, 0, 255).astype(np.uint8)
        return boosted
    raise ValueError(f"Unknown preprocess mode: {mode}")


def parse_raw_json(p: Path) -> Dict[str, Any]:
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def index_images_by_size(root: Path) -> Dict[Tuple[int, int], List[Path]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    idx: Dict[Tuple[int, int], List[Path]] = {}
    for p in root.iterdir():
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        idx.setdefault((w, h), []).append(p)
    return idx


def find_image_for_raw_entry(raw_dir: Path, name_hint: str, iw: int, ih: int, size_index: Dict[Tuple[int, int], List[Path]]) -> Path | None:
    # 1) exact path
    cand = raw_dir / name_hint
    if cand.exists():
        return cand
    # 2) case-insensitive stem equality
    stem = Path(name_hint).stem.lower()
    files = list(raw_dir.iterdir())
    for p in files:
        if p.stem.lower() == stem:
            return p
    # 3) fuzzy prefix/contains match
    for p in files:
        if p.stem.lower().startswith(stem) or stem in p.stem.lower():
            return p
    # 4) fallback by size
    if iw > 0 and ih > 0:
        cands = size_index.get((iw, ih), [])
        if len(cands) == 1:
            return cands[0]
        elif len(cands) > 1:
            # pick the first deterministically
            return sorted(cands)[0]
    return None


def build_from_raw(raw_dir: Path, out_root: Path, save_crops: bool, modes: List[str]) -> Dict[str, Any]:
    ensure_dir(out_root)
    det_items: List[Dict[str, Any]] = []
    recog_rows: List[Tuple[str, str, str]] = []

    crops_base = out_root / 'words'
    if save_crops:
        ensure_dir(crops_base)

    # collect all raw jsons
    json_files = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() == '.json'])
    size_index = index_images_by_size(raw_dir)
    for jp in json_files:
        data = parse_raw_json(jp)
        img_name = data.get('imagePath')
        iw = int(data.get('imageWidth') or 0)
        ih = int(data.get('imageHeight') or 0)
        if not img_name or iw <= 0 or ih <= 0:
            # try to read image to get size
            continue
        img_path = find_image_for_raw_entry(raw_dir, img_name, iw, ih, size_index)
        img = None
        if img_path is not None:
            img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] raw: could not resolve/read image for {jp.name} (hint: {img_name})")
            continue
        ih, iw = img.shape[:2]

        shapes = data.get('shapes', [])
        boxes_norm: List[List[float]] = []
        texts: List[str] = []
        for s in shapes:
            lbl = s.get('label', '') or ''
            pts = s.get('points') or []
            if len(pts) >= 2:
                (x1, y1), (x2, y2) = pts[0], pts[1]
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                # ensure order
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                boxes_norm.append(to_normalized_box_xyxy((x1, y1, x2, y2), iw, ih))
                texts.append(lbl)
                if save_crops:
                    crop = img[y1:y2, x1:x2]
                    if crop.size > 0:
                        base_name = f"{img_path.stem}_{x1}-{y1}-{x2}-{y2}"
                        for m in modes:
                            proc = apply_preprocess(crop, m)
                            out_rel = Path('words') / m / f"{base_name}.png"
                            out_abs = out_root / out_rel
                            ensure_dir(out_abs.parent)
                            cv2.imwrite(str(out_abs), proc)
                            recog_rows.append((str(out_rel).replace('\\', '/'), lbl, m))

        det_items.append({
            'image_path': str(img_path.relative_to(raw_dir.parent)).replace('\\', '/'),
            'width': iw,
            'height': ih,
            'boxes': boxes_norm,
            'transcriptions': texts,
        })

    manifest = {
        'images_root': str(raw_dir).replace('\\', '/'),
        'items': det_items,
    }

    with (out_root / 'detection_manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if save_crops and recog_rows:
        with (out_root / 'recognition_manifest.csv').open('w', encoding='utf-8') as f:
            f.write('image_path,text,preprocess\n')
            for relp, txt, m in recog_rows:
                safe_txt = txt.replace('\n', ' ').replace(',', ' ')
                f.write(f"{relp},{safe_txt},{m}\n")

    return manifest


def main():
    ap = argparse.ArgumentParser(description='Aggregate raw per-image JSONs into a DocTR detection manifest')
    ap.add_argument('--raw_dir', type=str, required=True, help='Directory containing images and per-image JSONs')
    ap.add_argument('--out', type=str, required=True, help='Output directory')
    ap.add_argument('--save_crops', action='store_true', help='Also save recognition crops and CSV manifest')
    ap.add_argument('--preprocess', type=str, default='original', help='Comma-separated: original,clahe,highboost')
    args = ap.parse_args()

    modes = [m.strip().lower() for m in args.preprocess.split(',') if m.strip()]
    build_from_raw(Path(args.raw_dir), Path(args.out), args.save_crops, modes)
    print('Done. Outputs written to:', args.out)


if __name__ == '__main__':
    main()
