import os

DATASET_ROOT = os.getenv("DATASET_ROOT", r"D:\AIC_DATA1_fromKG")

# Cấu trúc AIC 2025
KEYFRAMES_DIR = os.path.join(DATASET_ROOT, "frames")
CLIP_DIR      = os.path.join(DATASET_ROOT, "clip-features")
MAP_CSV_DIR   = os.path.join(DATASET_ROOT, "map-keyframes")

# File output cho app

IMAGE_MAP_JSON = os.path.join(DATASET_ROOT, "image_path.json")
FAISS_INDEX_BIN = os.path.join(DATASET_ROOT, "faiss_normal_ViT.bin")
FEATURES_ALL_NPY = os.path.join(DATASET_ROOT, "features_all.npy")# để tăng tốc image_search

# Xuất kết quả
RESULTS_XLSX      = os.path.join(os.getcwd(), "results.xlsx")

# Hiệu năng FAISS
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX", "flat")   # flat | ivf
FAISS_NLIST      = int(os.getenv("FAISS_NLIST", "4096"))
FAISS_NPROBE     = int(os.getenv("FAISS_NPROBE", "16"))

# GPU?  set USE_GPU=1 trước khi chạy để bật GPU nếu có
USE_GPU = os.getenv("USE_GPU", "0") == "1"
