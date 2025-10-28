"""

Video reference:
https://www.youtube.com/watch?v=06TE_U21FK4&t=233s

"""

import cv2
import mediapipe as mp
import numpy as np
import pygame


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Get Video feed from device 0
    cap = cv2.VideoCapture(0)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            
            ret, frame = cap.read()
            
            
            
            # Convert from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection from frame
            results = pose.process(image)
            
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
            
            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                # Get points to calculate angle
                left_shoulder_coords = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow_coords    = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist_coords    = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                angle = calculate_angle(left_shoulder_coords, left_elbow_coords, left_wrist_coords)
                
                # Visualize angle
                cv2.putText(image, str(int(angle)), 
                                tuple(np.multiply(left_elbow_coords, [width, height]).astype(int)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (99, 204, 43), 2, cv2.LINE_AA
                            )      

            except:
                pass
            
            
            
            # Draw detections to image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # landmark drawing spec
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # connection drawing spec
                                    )
            


            
            
            #cv2.imshow("mediapipe feed", frame)
            cv2.imshow("mediapipe feed", image)
            
            # if key pressed is q, quit
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        
    cap.release()
    cv2.destroyAllWindows()






def calculate_angle(first, mid, end):
    first = np.array(first)
    mid   = np.array(mid)
    end   = np.array(end)
    
    
    radians = np.arctan2(end[1]-mid[1], end[0]-mid[0]) - np.arctan2(first[1]-mid[1], first[0]-mid[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180:
        angle = 360-angle
    
    return angle


main()