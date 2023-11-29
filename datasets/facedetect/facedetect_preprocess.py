import cv2 as cv
import numpy as np
import csv
import os
import sys
import logging
import concurrent.futures
from mtcnn.detector import MtcnnDetector
from align_mtcnn import process


# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("facedetect_preprocess.log"),  # Log to a file
        logging.StreamHandler(),  # Log to the console
    ],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

aligned_csv_file_path = parent_dir + "/utkface/utkface.csv"
csv_file_path = parent_dir + "/utkface/utkfaceold.csv"
aligned_images_dir = parent_dir + "/utkface/aligned_images"


train_dir = os.path.join(parent_dir, aligned_images_dir, "train")
test_dir = os.path.join(parent_dir, aligned_images_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

if __name__ == "__main__":
    detector = MtcnnDetector()

    with open(csv_file_path, mode="r") as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)

        start_index = 12025

        # Create a new CSV file for aligned image data
        with open(aligned_csv_file_path, mode="a", newline="") as aligned_csv_file:
            csv_writer = csv.writer(aligned_csv_file)
            csv_writer.writerow(header)

            # Skip rows up to the start_index
            for _ in range(start_index):
                next(csv_reader)

            def process_row(row):
                age, image_path, data_type = row
                input_path = os.path.join(parent_dir, image_path)
                aligned_filename = os.path.basename(image_path).replace(
                    ".jpg", "_aligned.jpg"
                )
                output_path = os.path.join(
                    train_dir if data_type == "training" else test_dir, aligned_filename
                )

                raw = cv.imread(input_path)
                aligned_img_created = process(
                    detector=detector,
                    img=raw,
                    output_size=(112, 112),
                    output_path=output_path,
                )
                if aligned_img_created:
                    csv_writer.writerow(
                        [age, os.path.relpath(output_path)[3:], data_type]
                    )
                else:
                    logging.info(f"Unable to crop face from img: {image_path}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(process_row, csv_reader)
