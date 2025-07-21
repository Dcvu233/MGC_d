import os
import glob

import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

query_image_path = "./wash1.jpg"   # 查询图片
image_folder     = "./imgs"   # 图库文件夹，支持 jpg/png
use_gpu          = True                       # 如果有 CUDA，设为 True
# ========================================

# 设备设置
device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
print(f"[+] Using device: {device}")

# 加载 CLIP 模型 & 处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 1. 扫描图库并收集图片路径
exts = ("*.jpg", "*.jpeg", "*.png")
image_paths = []
for ext in exts:
    image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
if not image_paths:
    raise ValueError(f"No images found in {image_folder}")

# 2. 对图库图片计算特征并构建 FAISS 索引
print("[+] Encoding gallery images ...")
feats = []
for p in image_paths:
    img = Image.open(p).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)   # L2 归一化
    feats.append(emb.cpu().numpy())
feats = np.vstack(feats).astype("float32")

index = faiss.IndexFlatL2(feats.shape[1])
index.add(feats)
print(f"[+] Indexed {len(image_paths)} images.")

# 3. 对查询图片计算特征并搜索最相似
print("[+] Encoding query image ...")
q_img = Image.open(query_image_path).convert("RGB")
q_in = processor(images=q_img, return_tensors="pt").to(device)
with torch.no_grad():
    q_emb = model.get_image_features(**q_in)
q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)
q_vec = q_emb.cpu().numpy().astype("float32")

D, I = index.search(q_vec, 1)  # top-1
best_idx = int(I[0, 0])
print(f"\n最相似图片: {image_paths[best_idx]}")
print(f"距离 (L2, 越小越相似): {D[0, 0]:.4f}")
