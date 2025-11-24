import argparse
import cv2
import os
from pathlib import Path

def main():
  argument_parser = argparse.ArgumentParser()

  argument_parser.add_argument(
          '--name', default=None, type=str,
          help='name of the dataset output')

  args = argument_parser.parse_args()

  if args.name == None:
    exit()

  root_dir = Path(__file__).resolve().parent.parent

  output_folder = os.path.join(root_dir, "dataset", "train", args.name)

  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
    if key == ord('s'):
      image_filename = os.path.join(output_folder, f"image_{frame_count:04d}.jpg")
      cv2.imwrite(image_filename, frame)
      print(f"Image saved: {image_filename}")
      frame_count += 1

    elif key == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()