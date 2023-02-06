import cv2 
import numpy as np 

if __name__  == "__main__":
    # Load the video or image
    cap = cv2.VideoCapture("sample_data/left_2.MP4")

    # Select the first frame
    ret, frame = cap.read()

    # r: The row (y-coordinate) of the top-left corner of the ROI bounding box.
    # h: The height of the ROI bounding box.
    # c: The column (x-coordinate) of the top-left corner of the ROI bounding box.
    # w: The width of the ROI bounding box.

    # Define the initial position of the first hand
    r1, h1, c1, w1 = 932, 1228, 1483, 1540
    track_window1 = (c1, r1, w1, h1)

    # Define the initial position of the second hand
    r2, h2, c2, w2 = 1207, 1979-1207, 3100, 3840-3262 
    track_window2 = (c2, r2, w2, h2)

    # Create a region of interest for the first hand to track
    roi1 = frame[r1:r1+h1, c1:c1+w1]
    hsv_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_roi1, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist1 = cv2.calcHist([hsv_roi1], [0], mask1, [180], [0, 180])
    cv2.normalize(roi_hist1, roi_hist1, 0, 255, cv2.NORM_MINMAX)

    # Create a region of interest for the second hand to track
    roi2 = frame[r2:r2+h2, c2:c2+w2]
    hsv_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv_roi2, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist2 = cv2.calcHist([hsv_roi2], [0], mask2, [180], [0, 180])
    cv2.normalize(roi_hist2, roi_hist2, 0, 255, cv2.NORM_MINMAX)

    # Set up the termination criteria for the KLT tracker
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 640, 480)

    # Create a folder to store the output frames
    if not os.path.exists("custom/klt_frames"):
        os.makedirs("custom/klt_frames")

    # Start the video loop
    while True:
        ret, frame = cap.read()
        if ret == True:
            # Convert the current frame to HSV
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Calculate the back projection for the first hand
            dst1 = cv2.calcBackProject([hsv], [0], roi_hist1, [0, 180], 1)
            ret1, track_window1 = cv2.meanShift(dst1, track_window1, term_crit)
            x1, y1, w1, h1 = track_window1
            img2 = cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), 255, 2)
            
            # Calculate the back projection for the second hand
            dst2 = cv2.calcBackProject([hsv], [0], roi_hist2, [0, 180], 1)
            ret2, track_window2 = cv2.meanShift(dst2, track_window2, term_crit)
            x2, y2, w2, h2 = track_window2
            img2 = cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2), (0,255,0), 2)
            
            # Save the updated frame to the folder "klt_tracker_frames"
            frame_count_str = str(frame_count-1).zfill(4)
            cv2.imwrite("custom/klt_frames/{}.jpg".format(frame_count_str), img2)
            frame_count += 1
            
            # Display the updated frame
            cv2.imshow("Tracking", img2)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # cmd = "ffmpeg -r 30 -i custom/klt_frames/%04d.jpg -vcodec libx264 -pix_fmt yuv420p -y custom/klt_multi.mp4"
    # os.system(cmd)
    # cmd  = "rm -r custom/klt_frames"
    # os.system(cmd)