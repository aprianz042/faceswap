import os
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(320, 320))

swapper = get_model('./inswapper_128.onnx',
                    download=False,
                    providers=['CPUExecutionProvider'])

source_img = cv2.imread("source.jpg")
source_face = app.get(source_img)[0]

target_folder = "targets"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(target_folder):
    path = os.path.join(target_folder, file)
    target_img = cv2.imread(path)
    faces = app.get(target_img)

    if len(faces) == 0:
        continue

    result = swapper.get(target_img, faces[0], source_face, paste_back=True)
    cv2.imwrite(os.path.join(output_folder, file), result)

print("Selesai generate 7 ekspresi")