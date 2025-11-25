from pathlib import Path
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

root_dir = Path(__file__).resolve().parent.parent

MODEL_PATH = os.path.join(root_dir, "model", "face_classifier.h5")
TAGS_PATH = os.path.join(root_dir, "model", "tags.json")

try:
    classifier_model = load_model(MODEL_PATH)
    print(f"Modelo cargado desde: {MODEL_PATH}")
except Exception as e:
    classifier_model = None
    print(f"Error al cargar el modelo .h5: {e}")
    exit()

# Lista de nombres/etiquetas
CLASS_NAMES = None

with open(TAGS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

IMG_SIZE = (224, 224)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

# Clasifica la cara
def detect_and_classify_face(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    return faces


def main():
    if not classifier_model:
        print("El clasificador no se pudo cargar. Solo se hará la detección de caras.")

    while True:
        result, video_frame = video_capture.read()

        if result is False:
            break

        faces = detect_and_classify_face(video_frame)

        for x, y, w, h in faces:
            # Dibujar el rectángulo inicial
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            # Recortar la región de interés (ROI)
            face_roi = video_frame[y : y + h, x : x + w]

            # Preprocesamiento de la cara para la entrada del modelo
            # (Ajustar tamaño y normalizar)
            try:
                processed_face = cv2.resize(face_roi, IMG_SIZE)
                # Convertir a array de numpy y agregar la dimensión de batch (1, H, W, C)
                processed_face = np.expand_dims(processed_face, axis=0)
                # Normalizar (si tu modelo lo requiere, e.g., / 255.0)
                processed_face = processed_face / 255.0

                # Realizar la Predicción
                predictions = classifier_model.predict(processed_face)
                # Obtener el índice de la clase con la probabilidad más alta
                predicted_class_index = np.argmax(predictions)
                # Obtener la etiqueta (nombre)
                label = CLASS_NAMES[predicted_class_index]

                # obtener la confianza
                confidence = predictions[0][predicted_class_index]

                # Formatear el texto de la etiqueta
                label_text = f"{label} ({confidence:.2f})"

            except Exception as e:
                print(f"Error en el preprocesamiento o predicción: {e}")
                label_text = "Error de Clasificación"

            # Dibujar la Etiqueta sobre el Recuadro
            # La posición es un poco encima del recuadro
            cv2.putText(
                video_frame,
                label_text,
                (x, y - 10),  # Posición (x, y) - 10 para estar un poco más alto
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # Tamaño de la fuente
                (0, 255, 0),  # Color (verde)
                2,  # Grosor
            )
            cv2.putText(
                video_frame,
                "presiona 'q' para salir",
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (200, 200, 200),
                2,
            )

        cv2.imshow("Reconocimiento Facial con Keras", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
