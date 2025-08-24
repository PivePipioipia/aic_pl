# trake_module.py
from __future__ import annotations
import os, re
from typing import List, Tuple, Dict

# NOTE: Sẽ dùng MyFaiss và DictImagePath từ app (truyền vào hàm)
# Quy ước: video_name = 'L21_V001.mp4' => folder 'L21_V001'

# ---------- Parse events ----------
def parse_events_from_text(text: str, max_events: int = 4) -> List[str]:
    """
    Tách các event từ mô tả. Hỗ trợ nhiều pattern thường gặp:
    - 'Event1: ... Event2: ...'
    - 'E1: ...; E2: ...'
    - '... | ... | ...'
    - mỗi dòng một event
    """
    if not text:
        return []

    # 1) Tách theo "Event\d+:" hay "E\d+:"
    parts = re.split(r'(?:^|\s)(?:Event|E)\s*\d+\s*:\s*', text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts[:max_events]

    # 2) Tách theo ký tự phân cách thường dùng
    for sep in ['|', ';', '->', '→']:
        if sep in text:
            items = [s.strip() for s in text.split(sep)]
            items = [s for s in items if s]
            if len(items) >= 2:
                return items[:max_events]

    # 3) Mỗi dòng là một event
    lines = [l.strip("- \t") for l in text.splitlines() if l.strip()]
    if len(lines) >= 2:
        return lines[:max_events]

    # 4) Nếu không tách được, coi toàn bộ là 1 event
    return [text.strip()][:max_events]

# ---------- Helper ----------
def _folder_from_video(video_name: str) -> str:
    # 'L21_V001.mp4' -> 'L21_V001'
    return os.path.splitext(os.path.basename(video_name))[0]

def _frame_idx_from_relpath(rel_path: str) -> int:
    # 'L21_V001/0123.jpg' -> 123
    base = os.path.splitext(os.path.basename(rel_path))[0]
    digits = ''.join(ch for ch in base if ch.isdigit())
    try:
        return int(digits)
    except:
        return 0

# ---------- Core alignment using FAISS search ----------
def align_events(
    text_query: str,
    video_name: str,
    myfaiss,                      # instance Myfaiss
    dict_image_path: Dict[int,str],
    k_per_event: int = 200
) -> List[int]:
    """
    Trả về list frame_idx cho các event theo thứ tự thời gian trong video đã chọn.
    Chiến lược:
      - Tách event descriptions từ text_query.
      - Với mỗi event: FAISS text_search(event, k) -> lọc các keyframes thuộc folder video.
      - Chọn frame có điểm cao & đảm bảo frame tăng dần (monotonic).
    """
    folder = _folder_from_video(video_name)
    events = parse_events_from_text(text_query, max_events=4)
    if not events:
        return []

    chosen_frames: List[int] = []
    last_frame = -1

    for ev in events:
        # search top-K toàn corpus
        _, ids, _, paths = myfaiss.text_search(ev, k=k_per_event)
        # filter về đúng video folder
        cand: List[Tuple[int,int,str]] = []  # (id, frame_idx, rel_path)
        for i, p in zip(ids, paths):
            p = str(p).replace("\\","/")
            # p có dạng 'L21_V001/0000.jpg'
            if p.startswith(folder + "/"):
                cand.append((int(i), _frame_idx_from_relpath(p), p))
        # sort theo similarity đã có (giữ thứ tự hiện tại), rồi enforce thứ tự thời gian
        selected = None
        for cid, fidx, rp in cand:
            if fidx > last_frame:
                selected = fidx
                break
        # nếu không có frame lớn hơn last_frame (hiếm), lấy frame đầu tiên
        if selected is None and cand:
            selected = cand[0][1]

        if selected is not None:
            chosen_frames.append(selected)
            last_frame = selected
        else:
            # không có ứng viên trong video (event này bỏ trống)
            chosen_frames.append(last_frame if last_frame >= 0 else 0)

    return chosen_frames

# ---------- (Optional) Shot detection via TransNetV2 ----------
# Bạn có thể tích hợp thật sự TransNetV2 tại đây, ví dụ:
# def detect_shots(video_path: str) -> List[Tuple[int,int]]:
#     try:
#         from transnetv2 import TransNetV2
#         # TODO: load model + run inference -> trả về list (start_frame, end_frame)
#     except Exception:
#         return []
# Sau đó, trong align_events() bạn có thể giới hạn candidate vào các shot phù hợp để nhanh & chính xác hơn.
