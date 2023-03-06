import cv2
import numpy as np
import os
import pickle
from tqdm import tqdm

# Path to the folder containing the images
img_folder = 'hand_data/frames/undistorted_scene_sampled'

# Path to the folder to save the extrinsic matrices
extrinsic_folder = "feature_matching/extrinsics"

# Load the images
img_list = []
for filename in os.listdir(img_folder):
    if filename.endswith('.jpg'):
        img_list.append(os.path.join(img_folder, filename))
img_list = sorted(img_list)

# Create SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# Extract SIFT features and compute descriptors for each scene images
keypoints_list = []
descriptors_list = []
for img_path in tqdm(img_list, desc="Extracting SIFT features and computing descriptors"):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    keypoints_list.append(keypoints)
    descriptors_list.append(descriptors)

for filename in sorted(os.listdir("hand_data/frames/left_1")): 
    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # Detect SIFT features and compute descriptors for the single image
    single_img = cv2.imread("hand_data/frames/left_1/"+ filename , cv2.IMREAD_GRAYSCALE)
    single_keypoints, single_descriptors = sift.detectAndCompute(single_img, None)

    # Find the matches between the single image and each of the other images
    bf = cv2.BFMatcher()
    matches_list = []
    for descriptors2 in descriptors_list:
        matches = bf.knnMatch(single_descriptors, descriptors2, k=2)
        matches_list.append(matches)

    # Apply the distance ratio test to filter out the good matches
    good_matches_list = []
    for matches in matches_list:
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)
        good_matches_list.append(good_matches)

    # Use RANSAC to filter out the outliers and estimate the essential matrix and the rotation and translation vectors for each pair of images
    for i, good_matches in enumerate(tqdm(good_matches_list, desc="Estimating extrinsic matrices")):
        src_points = np.float32([single_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints_list[i][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        essential_matrix, mask = cv2.findEssentialMat(src_points, dst_points, focal=1.0, pp=(0,0), method=cv2.RANSAC, prob=0.999, threshold=3.0)
        _, R, t, _ = cv2.recoverPose(essential_matrix, src_points, dst_points)

    # Save the extrinsic matrix as a pkl file
    extrinsic_matrix = np.hstack((R, t))
    extrinsic_file = os.path.join(extrinsic_folder, "extrinsic_" + filename + ".pkl")
    with open(extrinsic_file, 'wb') as f:
        pickle.dump(extrinsic_matrix, f)

    print("Extrinsic matrice saved for: \t" + filename)