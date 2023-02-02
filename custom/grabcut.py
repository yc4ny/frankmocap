import numpy as np
import cv2

# Load the input image
img = cv2.imread('mocap_output/rendered/00000.jpg')

# Define the region of interest (ROI)
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (0, 0, 3820, 2160)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Create a binary mask
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Multiply the image with the mask to get the segmented hand
segmented_hand = img * mask2[:, :, np.newaxis]

# Display the output
# cv2.imshow("GrabCut", segmented_hand)
cv2.imwrite("grabcut.jpg", segmented_hand)
# cv2.waitKey(0)
# cv2.destroyAllWindows()