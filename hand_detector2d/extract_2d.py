import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def extract_2d(video_file):

    if not os.path.exists("hand_detector2d/output_pkl"):
        os.makedirs("hand_detector2d/output_pkl")
    if not os.path.exists("hand_detector2d/output_jpg"):
        os.makedirs("hand_detector2d/output_jpg")

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        cap = cv2.VideoCapture(video_file)
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames)):
            # Read a frame from the video file
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame with MediaPipe Hands
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Create a list to hold the hand joint locations
            hand_joints = []

            # Loop through each detected hand and append the joint locations to the list
            annotated_frame = frame.copy()
            if results.multi_hand_landmarks is not None:
                for hand_landmarks in results.multi_hand_landmarks:
                    joint_list = []
                    for landmark in hand_landmarks.landmark:
                        joint_list.append([landmark.x, landmark.y, landmark.z])
                    hand_joints.append(np.array(joint_list))

                # Save the hand joint locations to a pickle file
                output_path = f"hand_detector2d/output_pkl/pred_joint2d_{frame_count+1:05}.pkl"
                with open(output_path, 'wb') as f:
                    pickle.dump(hand_joints, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Visualize the hand landmarks on the frame
                image_height, image_width, _ = frame.shape
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            
            # Save the annotated frame to an image file
            output_path = f"hand_detector2d/output_jpg/{frame_count+1:05}.jpg"
            cv2.imwrite(output_path, annotated_frame)

            frame_count += 1

        cap.release()
        cmd = "ffmpeg -r 30 -i hand_detector2d/output_jpg/%05d.jpg -vcodec libx264 -pix_fmt yuv420p -y hand_detector2d/vis.mp4"
        os.system(cmd)


if __name__ == "__main__":
    extract_2d("hand_data/right_1.mp4")
