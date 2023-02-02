import cv2
import numpy as np
import os

# Load the video and create a ROI for the hand
cap = cv2.VideoCapture('sample_data/left_2.MP4')
_, frame = cap.read()

# r: The row (y-coordinate) of the top-left corner of the ROI bounding box.
# h: The height of the ROI bounding box.
# c: The column (x-coordinate) of the top-left corner of the ROI bounding box.
# w: The width of the ROI bounding box.

r, h, c, w = 836, 2160-836, 1473, 3840-1473
track_window = (c, r, w, h)

# Create a mask and compute the histogram of the hand
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Set up the termination criteria and start the tracking loop
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 640, 480)

# Create a folder to store the output frames
if not os.path.exists("custom/camshift_frames"):
    os.makedirs("custom/camshift_frames")

while True:
    # Read the next frame
    _, frame = cap.read()
    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Compute the backprojection of the histogram
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply the CamShift algorithm
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw the tracking window on the frame
    x, y, w, h = track_window
    img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Save the updated frame to the folder "klt_tracker_frames"
    frame_count_str = str(frame_count-1).zfill(4)
    cv2.imwrite("custom/camshift_frames/{}.jpg".format(frame_count_str), img2)
    frame_count += 1
    # Display the output frame
    cv2.imshow("Tracking", img2)

    # Exit the loop if the "q" key is pressed
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()

cmd = "ffmpeg -r 30 -i custom/camshift_frames/%04d.jpg -vcodec libx264 -pix_fmt yuv420p -y custom/camshift.mp4"
os.system(cmd)
cmd  = "rm -r custom/camshift_frames"
os.system(cmd)
