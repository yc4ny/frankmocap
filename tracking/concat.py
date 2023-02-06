import cv2
import os
from tqdm import tqdm

def concat_vertically(image1, image2):
    # concatenate the two images vertically
    return cv2.vconcat([image1, image2])

# directory containing the first set of images
folder_path1 = "mocap_output/bbox_rendered"

# directory containing the second set of images
folder_path2 = "mocap_output/original_rendered"

# directory to store the concatenated images
output_folder = "mocap_output/concat"

# list of filenames in folder1
filenames = [filename for filename in os.listdir(folder_path1) if filename.endswith(".jpg")]

# use tqdm to show a progress bar
for filename in tqdm(filenames, desc="Concatenating Images"):
    # read the first image
    image1 = cv2.imread(os.path.join(folder_path1, filename))

    # read the second image
    image2 = cv2.imread(os.path.join(folder_path2, filename))

    # concatenate the two images vertically
    result = concat_vertically(image1, image2)

    # save the result image to the output folder
    cv2.imwrite(os.path.join(output_folder, filename), result)