# scripts/build_features_and_index.py
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import faiss
import torch
import clip
from pathlib import Path
import config

def load_image_list():
    with open(config.IMAGE_MAP_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Dict key là số -> value là đường dẫn tương đối tới KEYFRAMES_DIR
    # Trả về list (theo thứ tự id tăng dần)
    items = sorted(((int(k), v) for k, v in raw.items()), key=lambda x: x[0])
    rel_paths = [v for _, v in items]
    abs_paths = [str(Path(config.KEYFRAMES_DIR) / p) for p in rel_paths]
    return rel_paths, abs_paths

def build_clip_features(abs_paths, device):
    # load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)  # giữ backbone giống app
    model.eval()

    feats = []
    for p in tqdm(abs_paths, desc="Encoding CLIP"):
        img = Image.open(p).convert("RGB")
        with torch.no_grad():
            x = preprocess(img).unsqueeze(0).to(device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)  # chuẩn hoá để dùng cosine
            feats.append(f.cpu().numpy().astype(np.float32))
    feats = np.concatenate(feats, axis=0)  # (N, D)
    return feats

def build_faiss_index(feats: np.ndarray):
    d = feats.shape[1]
    index_type = config.FAISS_INDEX_TYPE.lower()
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)  # cosine (do feats đã chuẩn hoá)
        index.add(feats)
    elif index_type == "ivf":
        # IVF với cosine: dùng inner product
        nlist = config.FAISS_NLIST
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        # train với một phần dữ liệu
        train_size = min(100_000, feats.shape[0])
        random_ids = np.random.choice(feats.shape[0], size=train_size, replace=False)
        index.train(feats[random_ids])
        index.add(feats)
        index.nprobe = config.FAISS_NPROBE
    else:
        raise ValueError(f"Unknown FAISS_INDEX_TYPE: {config.FAISS_INDEX_TYPE}")
    return index

def main():
    device = "cuda" if config.USE_GPU and torch.cuda.is_available() else "cpu"
    rel_paths, abs_paths = load_image_list()
    if len(abs_paths) == 0:
        raise RuntimeError("Không tìm thấy ảnh nào. Kiểm tra KEYFRAMES_DIR và image_path.json.")

    feats = build_clip_features(abs_paths, device=device)
    np.save(config.FEATURES_ALL_NPY, feats)
    print(f"✅ Lưu features vào {config.FEATURES_ALL_NPY} – shape {feats.shape}")

    index = build_faiss_index(feats)
    faiss.write_index(index, config.FAISS_INDEX_BIN)
    print(f"✅ Lưu FAISS index vào {config.FAISS_INDEX_BIN}")

if __name__ == "__main__":
    main()
