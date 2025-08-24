from PIL import Image
import faiss
import numpy as np
import clip
import os
import torch
from langdetect import detect

class Myfaiss:
    def __init__(self, bin_file: str, id2img_fps, device, translater,
                 clip_backbone="ViT-B/32", features_path=None):
        self.index = faiss.read_index(bin_file)
        self.id2img_fps = id2img_fps
        self.device = device
        self.model, self.preprocess = clip.load(clip_backbone, device=device)
        self.translater = translater

        self.features = None
        if features_path and os.path.exists(features_path):
            self.features = np.load(features_path).astype(np.float32)

    def _search(self, query_feats, k):
        # Normalize để so xấp xỉ cosine
        faiss.normalize_L2(query_feats)
        scores, idx_image = self.index.search(query_feats, k=k)
        idx_image = idx_image.flatten()
        image_paths = [self.id2img_fps[i] for i in idx_image]
        return scores, idx_image, image_paths, image_paths

    def image_search(self, id_query, k):
        # Query từ id frame trong dataset (giữ hành vi cũ)
        if self.features is not None:
            query_feats = self.features[int(id_query)].reshape(1, -1)
        else:
            query_feats = self.index.reconstruct(int(id_query)).reshape(1, -1)
        return self._search(query_feats, k)

    def image_search_from_pil(self, pil_img: Image.Image, k: int):
        # Query từ ảnh PIL (upload/sketch)
        img_in = self.preprocess(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(img_in).float().cpu().numpy()
        return self._search(feats, k)

    def text_search(self, text, k):
        if detect(text) == 'vi':
            text = self.translater(text)
        text_tok = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tok).float().cpu().numpy()
        return self._search(text_features, k)
