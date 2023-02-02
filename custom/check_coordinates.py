import cv2
import argparse

# Parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True, help="Path to the input image")
args = parser.parse_args()

# Load the input image
img = cv2.imread(args.image_path)

# Get the height and width of the image
height, width = img.shape[:2]

# Calculate the aspect ratio of the image
aspect_ratio = float(height) / float(width)

# Define the window size based on the aspect ratio
window_width = 800
window_height = int(window_width * aspect_ratio)

# Define a callback function for the mouse event
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("X:", x, "Y:", y)

# Create a named window with fixed size to display the image
cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Input Image", window_width, window_height)

# Set the mouse callback function for the named window
cv2.setMouseCallback("Input Image", click_event)

# Display the input image
cv2.imshow("Input Image", img)

# Wait for user to click on the image
cv2.waitKey(0)

# Destroy all windows
cv2.destroyAllWindows()