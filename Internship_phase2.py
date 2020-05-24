# The idea is retrived from:
# https://www.freecodecamp.org/news/smilfie-auto-capture-selfies-by-detecting-a-smile-using-opencv-and-python-8c5cfb6ec197/

from scipy.spatial import distance as dist
from imutils.video import VideoStream, FPS
from imutils import face_utils
import imutils
import numpy as np
import time
import dlib
import cv2
from statistics import mean

# You can adjust these parameters to best fit your facial expression. Find printing statement at the end of the code
EAR_ratio_treshhold = 0.18 # My eyes are pretty narrow when I smile, so the treshhold is a bit lower than ususal. 
MAR_ratio_treshhold = 0.36
jaw_ratio_treshhold = 2
open_mouth_ratio_treshhold = 2.5

def MAR_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar=avg/D
    return mar

def open_mouth_ratio(mouth):
    # We will measure distance between all exterior points and compare it with a circle. 
    # If the ratio is close to pi or 3, then the mouth is open and we should not calculate MAR
    circumference = sum([dist.euclidean(mouth[i], mouth[i + 1]) for i in range(12)])
    diameter = dist.euclidean(mouth[0], mouth[16])
    return circumference/diameter

def EAR_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def JAW_ratio(jaw):
    # Used to determine if viewer's head is strenthened too much up 
    # Testing on my own, I found that if the ratio is below 2, the head is too high and the program should not take a photo
    length = sum([dist.euclidean(jaw[i], jaw[i + 1]) for i in range(16)])
    wideness = dist.euclidean(jaw[0], jaw[16])
    ratio = length / wideness
    return ratio

frame_count = 0
total_frames = 0

cont_color = (0, 255, 0)
text_color = (0, 0, 255)

shape_predictor = "shape_predictor_68_face_landmarks.dat" 
detector        = dlib.get_frontal_face_detector()
predictor       = dlib.shape_predictor(shape_predictor)

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

print("Starting video stream thread...")
camera = VideoStream(0).start()
fileStream = False

while True:
    frame = camera.read()
    frame = imutils.resize(frame, width=700)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        mouth     = shape[mStart:mEnd]
        MAR       = MAR_ratio(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, cont_color, 1)
        
        left_eye       = shape[lStart:lEnd]
        right_eye      = shape[rStart:rEnd]
        left_eye_EAR   = EAR_ratio(left_eye)
        right_eye_EAR  = EAR_ratio(right_eye)
        left_eye_hull  = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, cont_color, 1)
        cv2.drawContours(frame, [right_eye_hull], -1, cont_color, 1)
        
        jaw       = shape[jStart:jEnd]
        jaw_ratio = JAW_ratio(jaw)
        jawHull   = cv2.convexHull(jaw)
        cv2.drawContours(frame, [jawHull], -1, cont_color, 1)
        
        is_head_position = jaw_ratio > jaw_ratio_treshhold
        is_mouth_round   = open_mouth_ratio(mouth) > open_mouth_ratio_treshhold
        is_MAR           = MAR > MAR_ratio_treshhold
        is_lEAR          = left_eye_EAR > EAR_ratio_treshhold
        is_rEAR          =  right_eye_EAR > EAR_ratio_treshhold
        
        smile_text = "Smile!" if not is_MAR or is_mouth_round else ""
        eyes_text = "Open your eyes!" if not is_lEAR and not is_rEAR else ""
        head_text = "Look straight at the camera" if not is_head_position else ""
        
        if is_head_position and not is_mouth_round and is_MAR and is_lEAR and is_rEAR: 
            frame_count += 1
            if frame_count >= 15:
                total_frames += 1
                frame    = camera.read()
                img_name = f"SmilingPicture_{total_frames}.png"
                cv2.imwrite(img_name, frame)
                print(f"{img_name} written!")
                frame_count = 0
                
        cv2.putText(frame, smile_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(frame, eyes_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(frame, head_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # Uncomment the next line if you want to tune the treshhold parameters
        # print(f"MAR: {round(MAR, 3)} | lEAR: {round(left_eye_EAR, 3)} | rEAR: {round(right_eye_EAR, 3)}")
        
    cv2.imshow("Smile Detector", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
camera.stop()

