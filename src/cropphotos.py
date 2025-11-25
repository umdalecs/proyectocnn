import os
import cv2
from pathlib import Path

DATASET_DIR = os.path.join(Path(__file__).resolve().parent.parent, "dataset", "train")


def create_face_dataset(input_dir):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    for dirpath, _, filenames in os.walk(input_dir):
        for file in filenames:
            image_path = os.path.join(dirpath, file)

            img = cv2.imread(image_path)
            if img is None:
                continue

            faces = face_cascade.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
                (x, y, w, h) = faces[0]

                p = 10
                x = max(0, x - p)
                y = max(0, y - p)
                w = min(img.shape[1] - x, w + 2 * p)
                h = min(img.shape[0] - y, h + 2 * p)

                face_crop = img[y : y + h, x : x + w]

                cv2.imwrite(image_path, face_crop)
            else:
              os.remove(image_path)


create_face_dataset(DATASET_DIR)
