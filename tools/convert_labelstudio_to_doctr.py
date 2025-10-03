import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def ls_percent_to_abs_bbox(value: Dict[str, Any], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert Label Studio percent-based bbox to absolute pixel (xmin, ymin, xmax, ymax).

    LS stores x,y as top-left percentages, and width,height as percentages.
    """
    x_perc = float(value["x"]) / 100.0
    y_perc = float(value["y"]) / 100.0
    w_perc = float(value["width"]) / 100.0
    h_perc = float(value["height"]) / 100.0

    xmin = int(round(x_perc * img_w))
    ymin = int(round(y_perc * img_h))
    xmax = int(round((x_perc + w_perc) * img_w))
    ymax = int(round((y_perc + h_perc) * img_h))

    # clip
    xmin = max(0, min(xmin, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    xmax = max(0, min(max(xmax, xmin + 1), img_w))
    ymax = max(0, min(max(ymax, ymin + 1), img_h))
    return xmin, ymin, xmax, ymax


def to_normalized_box(box: Tuple[int, int, int, int], img_w: int, img_h: int) -> List[float]:
    xmin, ymin, xmax, ymax = box
    return [xmin / img_w, ymin / img_h, xmax / img_w, ymax / img_h]


def apply_preprocess(img: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "original":
        return img
    if mode == "clahe":
        # Convert to LAB, apply CLAHE on L-channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    if mode == "highboost":
        # High-boost filtering: I_hb = I + alpha * (I - blur(I)) where alpha>0
        alpha = 1.5
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        mask = cv2.subtract(img, blurred)
        boosted = cv2.add(img, cv2.multiply(mask, np.array([alpha], dtype=np.float32)))
        boosted = np.clip(boosted, 0, 255).astype(np.uint8)
        return boosted
    raise ValueError(f"Unknown preprocess mode: {mode}")


def parse_labelstudio(ls_json_path: Path) -> List[Dict[str, Any]]:
    with ls_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level list in Label Studio export JSON")
    return data


def extract_image_relpath(task: Dict[str, Any]) -> str:
    # Try classic Label Studio under task["data"]
    data = task.get("data", {})
    for k in ("image", "img", "fp", "path"):
        if k in data:
            return data[k]
    # Try compact schema top-level
    for k in ("image", "image_path", "img", "path"):
        if k in task:
            return task[k]
    return ""


def index_images_by_size(images_root: Path) -> Dict[Tuple[int, int], List[Path]]:
    """Scan images_root for images and index by (width, height)."""
    index: Dict[Tuple[int, int], List[Path]] = {}
    # Accept common extensions
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in images_root.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        index.setdefault((w, h), []).append(p)
    return index


def extract_pairs(task: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Return list of (bbox_result, text_result) paired by shared id when possible.

    We handle two cases:
    - Paired by identical "id" between bbox and textarea results (common in LS setups)
    - Otherwise, fall back to order: zip bboxes with textareas in sequence
    """
    anns = task.get("annotations", [])
    if not anns:
        return []
    res = anns[0].get("result", [])

    bboxes = [r for r in res if r.get("type") == "rectanglelabels"]
    texts = [r for r in res if r.get("type") == "textarea"]

    # Try pair by id
    id_to_bbox = {r.get("id"): r for r in bboxes if r.get("id")}
    id_to_text = {r.get("id"): r for r in texts if r.get("id")}

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for rid, br in id_to_bbox.items():
        tr = id_to_text.get(rid)
        if tr is not None:
            pairs.append((br, tr))

    # If no pairs formed, fallback to sequential zipping
    if not pairs and bboxes and texts:
        for br, tr in zip(bboxes, texts):
            pairs.append((br, tr))

    return pairs


def extract_compact_boxes_and_texts(task: Dict[str, Any], iw: int, ih: int) -> Tuple[List[List[float]], List[str]]:
    """Handle compact schema where task has keys like 'image' and 'bbox': [...].
    Each element of bbox is expected to contain percent coords and possibly original_width/height.
    Texts may be provided in a parallel list under 'texts' or 'labels'. If absent, use empty strings.
    """
    boxes_norm: List[List[float]] = []
    texts: List[str] = []
    bboxes = task.get("bbox") or []
    # Possible parallel text arrays
    text_list = task.get("texts") or task.get("labels") or task.get("transcriptions") or []
    for idx, b in enumerate(bboxes):
        if not isinstance(b, dict):
            continue
        value = {
            "x": b.get("x", 0.0),
            "y": b.get("y", 0.0),
            "width": b.get("width", 0.0),
            "height": b.get("height", 0.0),
        }
        # Use actual image size iw, ih to convert percent -> abs as in LS
        box_abs = ls_percent_to_abs_bbox(value, iw, ih)
        boxes_norm.append(to_normalized_box(box_abs, iw, ih))
        t = ""
        if idx < len(text_list):
            t = text_list[idx] or ""
        texts.append(t)
    return boxes_norm, texts


def build_detection_manifest(ls_data: List[Dict[str, Any]], images_root: Path, out_root: Path,
                             save_crops: bool = False, crop_preproc: List[str] = None,
                             crop_size_limit: int = 4096, crop_format: str = "png") -> Dict[str, Any]:
    ensure_dir(out_root)
    det_items: List[Dict[str, Any]] = []

    crops_dir = out_root / "words"
    if save_crops:
        ensure_dir(crops_dir)

    recog_rows: List[Tuple[str, str, str]] = []  # (rel_path, text, preprocess)

    # Build a size index for fallback matching
    size_index = index_images_by_size(images_root)
    used_by_size: Dict[Tuple[int, int], set] = {k: set() for k in size_index.keys()}

    # Build a global sorted image list for sequential fallback
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    all_images = sorted([p for p in images_root.iterdir() if p.is_file() and p.suffix.lower() in exts])
    global_used = set()
    global_idx = 0

    for task in ls_data:
        # Determine image path
        rel = extract_image_relpath(task)
        # LS percentage coordinates always relate to original_width/height in results; but safer to read actual image
        # We'll locate image by task["data"]["image"] filename in images_root if rel is not absolute within the root.
        img_path = None
        if rel:
            # Use only basename to match inside images_root
            candidate = images_root / Path(rel).name
            if candidate.exists():
                img_path = candidate
        if img_path is None:
            # Fallback by (original_width, original_height) from any result
            anns = task.get("annotations", [])
            res = anns[0].get("result", []) if anns else []
            ow, oh = None, None
            for r in res:
                ow = r.get("original_width")
                oh = r.get("original_height")
                if ow and oh:
                    break
            if ow and oh:
                candidates = size_index.get((int(ow), int(oh)), [])
                if len(candidates) == 1:
                    img_path = candidates[0]
                    used_by_size[(int(ow), int(oh))].add(str(img_path))
                elif len(candidates) > 1:
                    # Prefer an unused candidate for this size
                    key = (int(ow), int(oh))
                    pool = [c for c in candidates if str(c) not in used_by_size[key]]
                    if pool:
                        img_path = pool[0]
                        used_by_size[key].add(str(img_path))
                    else:
                        # All used; fall back to first (may duplicate) but warn
                        img_path = candidates[0]
                        print(f"[WARN] Exhausted unique candidates for size {key}; reusing {img_path.name}.")
            if img_path is None:
                # Final fallback: assign next unused image in folder order
                while global_idx < len(all_images) and str(all_images[global_idx]) in global_used:
                    global_idx += 1
                if global_idx < len(all_images):
                    img_path = all_images[global_idx]
                    global_used.add(str(img_path))
                    global_idx += 1
                else:
                    print("[WARN] No images left to assign for a task; skipping.")
                    continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue
        ih, iw = img.shape[:2]

        # Determine format: classic LS (has annotations->result) or compact (has bbox list)
        boxes_norm: List[List[float]] = []
        texts: List[str] = []
        used_box_abs: List[Tuple[int, int, int, int]] = []

        anns = task.get("annotations")
        if isinstance(anns, list) and anns and isinstance(anns[0], dict) and isinstance(anns[0].get("result"), list):
            pairs = extract_pairs(task)
            for bbox_res, text_res in pairs:
                value = bbox_res.get("value", {})
                box_abs = ls_percent_to_abs_bbox(value, iw, ih)
                used_box_abs.append(box_abs)
                boxes_norm.append(to_normalized_box(box_abs, iw, ih))

                tval = text_res.get("value", {})
                arr = tval.get("text") or []
                text = arr[0] if arr else ""
                texts.append(text)
        elif isinstance(task.get("bbox"), list):
            # Compact schema
            boxes_norm, texts = extract_compact_boxes_and_texts(task, iw, ih)
            # Recompute abs boxes from norm for cropping
            for bn in boxes_norm:
                xmin = int(round(bn[0] * iw))
                ymin = int(round(bn[1] * ih))
                xmax = int(round(bn[2] * iw))
                ymax = int(round(bn[3] * ih))
                used_box_abs.append((xmin, ymin, xmax, ymax))
        else:
            # Unknown schema; skip
            continue

        # Save crops if requested
        if save_crops and used_box_abs:
            modes = crop_preproc or ["original"]
            for (xmin, ymin, xmax, ymax), text in zip(used_box_abs, texts or []):
                crop = img[ymin:ymax, xmin:xmax]
                if crop.size == 0:
                    continue
                if max(crop.shape[0], crop.shape[1]) > crop_size_limit:
                    scale = crop_size_limit / float(max(crop.shape[0], crop.shape[1]))
                    crop = cv2.resize(crop, (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
                base_name = f"{img_path.stem}_{xmin}-{ymin}-{xmax}-{ymax}"
                for m in modes:
                    proc = apply_preprocess(crop, m)
                    ext = ".jpg" if crop_format.lower() == "jpg" else ".png"
                    out_rel = Path("words") / m / f"{base_name}{ext}"
                    out_abs = out_root / out_rel
                    ensure_dir(out_abs.parent)
                    if ext == ".jpg":
                        cv2.imwrite(str(out_abs), proc, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    else:
                        cv2.imwrite(str(out_abs), proc)
                    # Even if text is empty, include a row
                    recog_rows.append((str(out_rel).replace("\\", "/"), text or "", m))

        det_items.append({
            "image_path": str(img_path.relative_to(images_root.parent)).replace("\\", "/"),
            "width": iw,
            "height": ih,
            "boxes": boxes_norm,  # [[xmin, ymin, xmax, ymax]] normalized
            "transcriptions": texts,
        })

    manifest = {
        "images_root": str(images_root).replace("\\", "/"),
        "items": det_items,
    }

    # Write detection manifest
    with (out_root / "detection_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Write recognition manifest if any
    if save_crops and recog_rows:
        rec_path = out_root / "recognition_manifest.csv"
        with rec_path.open("w", encoding="utf-8") as f:
            f.write("image_path,text,preprocess\n")
            for relp, txt, m in recog_rows:
                # Quote CSV minimally by replacing newlines/commas
                safe_txt = txt.replace("\n", " ").replace(",", " ")
                f.write(f"{relp},{safe_txt},{m}\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Convert Label Studio OCR annotations to manifests for DocTR-style training")
    parser.add_argument("--annotations", type=str, required=True, help="Path to Label Studio JSON export")
    parser.add_argument("--images_root", type=str, required=True, help="Directory containing the images referenced in annotations")
    parser.add_argument("--out", type=str, required=True, help="Output directory for manifests and optional crops")
    parser.add_argument("--save_crops", action="store_true", help="Also save word crops and a recognition CSV manifest")
    parser.add_argument("--preprocess", type=str, default="original", help="Comma-separated list of preprocessing variants: original,clahe,highboost")
    parser.add_argument("--crop_format", type=str, default="png", choices=["png", "jpg"], help="Format for saved word crops when --save_crops is used")

    args = parser.parse_args()

    ls_path = Path(args.annotations)
    images_root = Path(args.images_root)
    out_root = Path(args.out)

    if not ls_path.exists():
        raise SystemExit(f"Annotations not found: {ls_path}")
    if not images_root.exists():
        raise SystemExit(f"Images root not found: {images_root}")

    modes = [m.strip().lower() for m in args.preprocess.split(",") if m.strip()]

    data = parse_labelstudio(ls_path)
    build_detection_manifest(data, images_root, out_root, save_crops=args.save_crops, crop_preproc=modes, crop_format=args.crop_format)
    print(f"Done. Outputs written to: {out_root}")


if __name__ == "__main__":
    main()
