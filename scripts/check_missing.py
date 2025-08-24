# scripts/check_missing.py
import os
import glob
import config   # dùng config.py để lấy đường dẫn gốc

# lấy thư mục từ config
root = config.DATASET_ROOT
feat_dir = config.CLIP_DIR        # D:\AIC_DATA1_fromKG\clip-features
kf_dir   = config.KEYFRAMES_DIR   # D:\AIC_DATA1_fromKG\frames

# lấy tất cả file .npy trong clip-features
feat_files = glob.glob(os.path.join(feat_dir, "*.npy"))

missing = []
for f in feat_files:
    vid = os.path.splitext(os.path.basename(f))[0]   # tên video, vd L21_V001
    kf_subdir = os.path.join(kf_dir, vid)           # frames/L21_V001
    if not os.path.isdir(kf_subdir):
        missing.append(vid)

print(" Tổng số features:", len(feat_files))
if missing:
    print(" Videos có features nhưng chưa có keyframes:", missing)
else:
    print(" Không thiếu keyframes cho video nào")
