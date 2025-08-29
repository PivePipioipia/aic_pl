import os, json
import config

ALL_MAPS = {}

def load_all_maps():
    global ALL_MAPS
    if not os.path.exists(config.MAP_IDX_DIR):
        print(f"MAP_IDX_DIR không tồn tại: {config.MAP_IDX_DIR}")
        return {}

    for fname in os.listdir(config.MAP_IDX_DIR):
        if fname.endswith(".json"):
            fpath = os.path.join(config.MAP_IDX_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            video_name = fname.replace("map_", "").replace(".json", "")
            ALL_MAPS[video_name] = data

    print(f"Loaded {len(ALL_MAPS)} map files từ {config.MAP_IDX_DIR}")
    return ALL_MAPS


def keyframe_to_video_frame(rel_path: str):
    """
    Input:  'L21_V001/087.jpg'
    Output: ('L21_V001', 2678)
    """
    try:
        video, fname = rel_path.split("/", 1)
        mapping = ALL_MAPS.get(video, {})
        frame_idx = mapping.get(fname)
        return (video, frame_idx)
    except Exception as e:
        print("Mapping error:", e)
        return None


# load khi import
load_all_maps()
