import argparse
import cv2
import os
from pathlib import Path
import random


def recortar_cuadrada(img):
    alto, ancho, _ = img.shape

    nuevo_tamano = min(alto, ancho)

    x_inicio = (ancho - nuevo_tamano) // 2
    y_inicio = (alto - nuevo_tamano) // 2

    img_cuadrada = img[
        y_inicio : y_inicio + nuevo_tamano, x_inicio : x_inicio + nuevo_tamano
    ]

    return img_cuadrada


def main():
    flag = False
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        "--name", default=None, type=str, help="name of the dataset output"
    )

    args = argument_parser.parse_args()

    if args.name == None:
        print("No --name arg provided")
        exit()

    root_dir = Path(__file__).resolve().parent.parent

    output_folder_train = os.path.join(root_dir, "dataset", "train", args.name)
    output_folder_val = os.path.join(root_dir, "dataset", "val", args.name)

    os.makedirs(output_folder_train, exist_ok=True)
    os.makedirs(output_folder_val, exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video device")
        exit()

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.imshow("Live Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            flag = not flag

        if flag:
            frame_cuadrada = recortar_cuadrada(frame)

            if random.random() > 0.2:
                image_filename = os.path.join(
                    output_folder_train, f"image_{frame_count:04d}.png"
                )
            else:
                image_filename = os.path.join(
                    output_folder_val, f"image_{frame_count:04d}.png"
                )

            cv2.imwrite(image_filename, frame_cuadrada)
            print(f"Image saved: {image_filename}")
            frame_count += 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
