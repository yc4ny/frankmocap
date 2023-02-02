import cv2
import numpy as np
import os

cap = cv2.VideoCapture('sample_data/left_2.MP4')

# parameters for Farneback optical flow
params = dict(pyr_scale = 0.5,
              levels = 3,
              winsize = 15,
              iterations = 3,
              poly_n = 5,
              poly_sigma = 1.2,
              flags = 0)

# take the first frame of the video
ret, prev_frame = cap.read()

# convert the frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# initialize the bounding box
x, y, w, h = 1459, 859, 3003-1497, 2160-897 
cv2.namedWindow("Optical Flow Bounding Box", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Optical Flow Bounding Box", 640, 480)

# Create a folder to store the output frames
if not os.path.exists("custom/opticalflow_frames"):
    os.makedirs("custom/opticalflow_frames")

while True:
    ret, frame = cap.read()
    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # compute optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **params)

    # calculate the magnitude and angle of the optical flow
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # apply a threshold to the magnitude to eliminate noise
    threshold = np.mean(magnitude)
    magnitude[magnitude < threshold] = 0

    # get the average flow in the bounding box
    avg_flow = np.mean(magnitude[y:y+h, x:x+w])

    # update the bounding box based on the average flow
    x += int(avg_flow)
    y += int(avg_flow)

    # draw the bounding box on the frame
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Optical Flow Bounding Box", frame)
    # Save the updated frame to the folder "klt_tracker_frames"
    frame_count_str = str(frame_count-1).zfill(4)
    cv2.imwrite("custom/opticalflow_frames/{}.jpg".format(frame_count_str), frame)

    # update the previous frame and grayscale image
    prev_gray = gray.copy()

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

cmd = "ffmpeg -r 30 -i custom/opticalflow_frames/%04d.jpg -vcodec libx264 -pix_fmt yuv420p -y custom/opticalflow.mp4"
os.system(cmd)
cmd  = "rm -r custom/opticalflow_frames"
os.system(cmd)