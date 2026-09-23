import faiss
import numpy as np

d = 64
xb = np.random.random((1000, d)).astype("float32")
index = faiss.IndexFlatIP(d)
index.add(xb)
D, I = index.search(xb[:5], 3)
print("faiss version:", faiss.__version__)
print("search OK, neighbors shape:", I.shape)
