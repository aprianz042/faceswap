import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# =========================
# 1. Load Face Analysis (CPU)
# =========================
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))  # -1 = CPU

# =========================
# 2. Load Swap Model (Manual Path)
# =========================
swapper = get_model(
    './inswapper_128.onnx',
    download=False,
    providers=['CPUExecutionProvider']
)

# =========================
# 3. Load Images
# =========================
source_img = cv2.imread("putin.jpg")
target_img = cv2.imread("arab.jpg")

if source_img is None or target_img is None:
    print("Gambar tidak ditemukan!")
    exit()

# =========================
# 4. Detect Faces
# =========================
source_faces = app.get(source_img)
target_faces = app.get(target_img)

if len(source_faces) == 0:
    print("Wajah di source tidak terdeteksi!")
    exit()

if len(target_faces) == 0:
    print("Wajah di target tidak terdeteksi!")
    exit()

# Ambil wajah pertama
source_face = source_faces[0]
target_face = target_faces[0]

# =========================
# 5. Swap
# =========================
result = swapper.get(
    target_img,
    target_face,
    source_face,
    paste_back=True
)

# =========================
# 6. Save Result
# =========================
cv2.imwrite("swapped_result.jpg", result)

print("✅ Face swap selesai!")
print("Hasil tersimpan sebagai: swapped_result.jpg")