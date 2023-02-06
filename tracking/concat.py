import cv2
import os
from tqdm import tqdm
import os 

def concat_vertically(image1, image2):
    # concatenate the two images vertically
    return cv2.vconcat([image1, image2])

# directory containing the first set of images
folder_path1 = "mocap_output/bbox_rendered"

# directory containing the second set of images
folder_path2 = "mocap_output/original_rendered"

folder_path = "mocap_output/concat"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# directory to store the concatenated images
output_folder = "mocap_output/concat"

# list of filenames in folder1
filenames = [filename for filename in os.listdir(folder_path1) if filename.endswith(".jpg")]

font = cv2.FONT_HERSHEY_SIMPLEX
text = "Tracking"
font_scale = 7
thickness = 5
text_size = cv2.getTextSize(text, font, font_scale, thickness)
(width, height) = text_size[0]
text_color = (255, 255, 255)
text_outline_color = (255, 0, 0)

# use tqdm to show a progress bar
for filename in tqdm(sorted(filenames), desc="Concatenating Images"):
    # read the first image
    image1 = cv2.imread(os.path.join(folder_path1, filename))
    x = image1.shape[1] - width - 10
    y = image1.shape[0] - height - 10
    cv2.putText(image1, text, (x, y), font, font_scale, text_color, thickness)
    cv2.putText(image1, text, (x, y), font, font_scale, text_outline_color, thickness + 2)
    # read the second image
    image2 = cv2.imread(os.path.join(folder_path2, filename))
    x = image2.shape[1] - width - 10
    y = image2.shape[0] - height - 10
    cv2.putText(image2, "Original", (x, y), font, font_scale, text_color, thickness)
    cv2.putText(image2, "Original", (x, y), font, font_scale, text_outline_color, thickness + 2)
    # concatenate the two images vertically
    result = concat_vertically(image1, image2)

    # save the result image to the output folder
    cv2.imwrite(os.path.join(output_folder, filename), result)

cmd = "ffmpeg -r 30 -i mocap_output/concat/%05d.jpg -vcodec libx264 -pix_fmt yuv420p -y mocap_output/concat.mp4"
os.system(cmd)
cmd  = "rm -rf concat/*.jpg"
os.system(cmd)