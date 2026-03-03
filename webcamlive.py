import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# =========================
# Load face analysis (CPU)
# =========================
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(320, 320))  # kecil supaya lebih cepat

# =========================
# Load swap model (CPU)
# =========================
swapper = get_model(
    './inswapper_128.onnx',
    download=False,
    providers=['CPUExecutionProvider']
)

# =========================
# Load source face (wajah yang akan dipakai)
# =========================
source_img = cv2.imread("putin.jpg")
source_faces = app.get(source_img)

if len(source_faces) == 0:
    print("Wajah source tidak terdeteksi!")
    exit()

source_face = source_faces[0]

# =========================
# Start webcam
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam tidak bisa dibuka")
    exit()

print("Tekan Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        frame = swapper.get(frame, face, source_face, paste_back=True)

    cv2.imshow("Live Face Swap (CPU)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()