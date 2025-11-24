from pathlib import Path
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

root_dir = Path(__file__).resolve().parent.parent

MODEL_PATH = os.path.join(root_dir, "model","face_classifier.h5")
TAGS_PATH = os.path.join(root_dir, "dataset","train")

# 1. --- Configuración del Modelo de Clasificación (Keras .h5) ---
try:
    # Cargar el modelo entrenado
    classifier_model = load_model(MODEL_PATH)
    print(f"Modelo cargado desde: {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo .h5: {e}")
    # Si no puedes cargar el modelo, el programa no podrá clasificar
    classifier_model = None 

# Lista de nombres/etiquetas
# ¡IMPORTANTE! Asegúrate de que este orden coincida EXACTAMENTE con el orden 
# de las clases que usaste para entrenar tu modelo.
CLASS_NAMES = os.listdir(TAGS_PATH)
IMG_SIZE = (224, 224) # Reemplaza con el tamaño de entrada de tu modelo

# 2. --- Configuración del Detector de Caras (OpenCV Haar Cascade) ---
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 3. --- Captura de Video ---
video_capture = cv2.VideoCapture(0)

# 4. --- Función de Detección y Clasificación Modificada ---
def detect_and_classify_face(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    
    # Detección de caras (Haar Cascade)
    faces = face_classifier.detectMultiScale(
        gray_image, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(40, 40)
    )

    return faces

# 6. --- Función Principal (Bucle de Video) ---
def main():
    # ¡Asegúrate de que la ruta de tu modelo en MODEL_PATH sea correcta!
    if not classifier_model:
        print("El clasificador no se pudo cargar. Solo se hará la detección de caras.")
        
    while True:
        result, video_frame = video_capture.read()

        if result is False:
            break

        # Llamar a la nueva función
        faces = detect_and_classify_face(video_frame)
        
        for (x, y, w, h) in faces:
        # Dibujar el rectángulo inicial
            cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            if classifier_model:
                # Recortar la región de interés (ROI)
                face_roi = video_frame[y:y + h, x:x + w]
                
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
                    
                    # Opcional: obtener la confianza
                    confidence = predictions[0][predicted_class_index]
                    
                    # Formatear el texto de la etiqueta
                    label_text = f"{label} ({confidence:.2f})" 

                except Exception as e:
                    print(f"Error en el preprocesamiento o predicción: {e}")
                    label_text = "Error de Clasificación"
            else:
                label_text = "Modelo No Cargado"
            
        # 5. Dibujar la Etiqueta sobre el Recuadro
        # La posición es un poco encima del recuadro
        cv2.putText(
            video_frame, 
            label_text, 
            (x, y - 10), # Posición (x, y) - 10 para estar un poco más alto
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, # Tamaño de la fuente
            (0, 255, 0), # Color (verde)
            2 # Grosor
        )

        cv2.imshow(
            "Reconocimiento Facial con Keras", video_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()