import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input folder")
ap.add_argument("-o", "--output", required=True, help="path to output folder")
args = vars(ap.parse_args())

# Define the input and output folders
input_folder = args["input"]
output_folder = args["output"]

# Define the intrinsic parameters
fx = 1792.358412
fy = 1826.892351
cx = 1920.000000
cy = 1080.000000
k1 = -0.208485
k2 = 0.043618
p1 = -0.004888
p2 = -0.004355

# Define the intrinsic matrix
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

# Define the distortion coefficients
dist_coeffs = np.array([k1, k2, p1, p2])

# Make sure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of all the image files in the input folder
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')])

# Iterate over each image file and undistort it
for image_file in tqdm(image_files, desc="Undistorting images", unit="image"):
    # Load the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)

    # Undistort the image
    undistorted = cv2.undistort(image, K, dist_coeffs)

    # Save the undistorted image
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, undistorted)