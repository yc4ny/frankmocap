import os
import shutil
import argparse

# Define the command-line arguments
parser = argparse.ArgumentParser(description='Select every 5th image from a folder')
parser.add_argument('input', help='path to the input folder')
parser.add_argument('output', help='path to the output folder')
args = parser.parse_args()

# Make sure the input and output folders exist
if not os.path.exists(args.input):
    raise ValueError(f"Input folder '{args.input}' does not exist.")
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Get a list of all the image files in the input folder
image_files = [f for f in os.listdir(args.input) if f.endswith('.jpg') or f.endswith('.png')]

# Iterate over every 5th image and copy it to the output folder
for i, image_file in enumerate(image_files):
    if (i + 1) % 5 == 0:
        shutil.copy(os.path.join(args.input, image_file), args.output)