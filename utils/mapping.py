# utils/mapping.py
from __future__ import annotations
import os, csv, glob
from typing import Dict, Tuple, Optional
import config

# Trả về: dict[str_rel_path] -> (video_name.mp4, frame_idx:int)
_mapping: Dict[str, Tuple[str, int]] = {}
_loaded = False

# Các tên cột có thể gặp trong CSV map
CAND_COLS_KEYFRAME = ["keyframe", "keyframe_path", "image_path", "rel_path"]
CAND_COLS_VIDEO    = ["video", "video_id", "video_name", "video_file"]
CAND_COLS_FRAME    = ["frame", "frame_id", "frame_idx", "frame_index"]

def _first_hit(cols, header):
    for c in cols:
        if c in header:
            return c
    return None

def _load_from_csv_dir(dir_path: str):
    global _mapping
    csv_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    for f in csv_files:
        with open(f, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = [h.strip() for h in reader.fieldnames or []]
            kcol = _first_hit(CAND_COLS_KEYFRAME, header)
            vcol = _first_hit(CAND_COLS_VIDEO, header)
            fcol = _first_hit(CAND_COLS_FRAME, header)
            if not (kcol and vcol and fcol):
                # CSV không khớp định dạng kỳ vọng, bỏ qua file này
                continue
            for row in reader:
                key = str(row[kcol]).replace("\\", "/").strip()
                vid = str(row[vcol]).strip()
                try:
                    fidx = int(row[fcol])
                except:
                    # nếu không parse được số -> bỏ qua dòng này
                    continue
                if key:
                    _mapping[key] = (vid, fidx)

def _fallback_build_from_filename():
    """
    Nếu không có CSV map: suy luận
    'L21_V001/0000.jpg' -> ('L21_V001.mp4', frame_idx=0)
    """
    # đọc image_path.json để lấy tất cả rel_paths
    import json
    with open(config.IMAGE_MAP_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        items = [v for _, v in sorted([(int(k), v) for k, v in raw.items()], key=lambda x: x[0])]
    elif isinstance(raw, list):
        items = list(raw)
    else:
        items = []

    for rel in items:
        rel = str(rel).replace("\\", "/")
        parts = rel.split("/")
        if len(parts) >= 2:
            folder = parts[0]
            fname  = parts[-1]
            video  = f"{folder}.mp4"
            digits = "".join(ch for ch in os.path.splitext(fname)[0] if ch.isdigit())
            try:
                fidx = int(digits)
            except:
                fidx = 0
            _mapping[rel] = (video, fidx)

def ensure_loaded():
    global _loaded
    if _loaded:
        return
    if os.path.isdir(config.MAP_CSV_DIR):
        _load_from_csv_dir(config.MAP_CSV_DIR)
    if not _mapping:
        _fallback_build_from_filename()
    _loaded = True

def keyframe_to_video_frame(rel_path: str) -> Optional[Tuple[str, int]]:
    """
    rel_path: 'L21_V001/0000.jpg'
    -> ('L21_V001.mp4', 0) nếu có map; None nếu không tìm thấy.
    """
    ensure_loaded()
    rel = rel_path.replace("\\", "/").strip()
    return _mapping.get(rel)
