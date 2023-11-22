# Author: Abdur Rahman Mohammed

import os
import csv
import random
from math import ceil 

# set path for image paths
part_folders = ["part1", "part2", "part3"]

# create csv file with the dataset
csv_file_path = "utkface.csv"
image_path_prefix = "utkface"

# Function to split the data into training and testing sets
def split_data(data, train_percent):
    random.shuffle(data)
    split_index = ceil(len(data) * train_percent)
    return data[:split_index], data[split_index:]

with open(csv_file_path, mode = 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(['age', 'path', 'type'])

    for part_folder in part_folders:
        part_path = os.path.join(os.getcwd(), part_folder)
        files = os.listdir(part_path)

        # Generate data for each image
        data = []
        for file in files:
            if file.endswith('.jpg'):
                # Extract information from the filename
                parts = file.split('_')
                age = parts[0]
                path = os.path.join(image_path_prefix,part_folder, file)

                # Assign to training or testing set randomly
                data.append((age, path))

        # Split the data into training and testing sets
        train_data, test_data = split_data(data, train_percent=0.9)

        # Write data to CSV file
        for age, path in train_data:
            csv_writer.writerow([age, path, 'training'])

        for age, path in test_data:
            csv_writer.writerow([age, path, 'testing'])
