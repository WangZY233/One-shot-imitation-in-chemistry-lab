import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import cv2
import matplotlib.pyplot as plt
import os
import time
from pathlib import Path
from argparse import ArgumentParser
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def process_frame_results(rgb_image, detection_result):
    '''
    Annotate the image with the hand landmarks and handedness.
    Also process the results to get the average x and y coordinates of the hand landmarks in this frame.

    Mostly copied from the MediaPipe example. Kinda messy to be honest.

    Input:
    rgb_image: np.array. The image to be annotated.
    detection_result: mp.tasks.vision.HandLandmarkerResult. The detection result.

    Return:
    annotated_image: np.array. The annotated image.
    avg_landmark_x: dict. The average x coordinate of the hand landmarks in this frame. Both left and right hand.
    avg_landmark_y: dict. The average y coordinate of the hand landmarks in this frame. Both left and right hand.
    all_handedness: set. All handedness detected in this frame.
    '''
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    

    avg_landmark_x = {} # average x coordinate of the hand landmarks in this frame
    avg_landmark_y = {} # average y coordinate of the hand landmarks in this frame
    all_handedness = set() # all handedness detected in this frame
    Thumb_Tip = [0,0]
    Index_Finger_Tip = [0,0]
    
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList() # type: ignore
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks # type: ignore
        ])
        solutions.drawing_utils.draw_landmarks( # type: ignore
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS, # type: ignore
        solutions.drawing_styles.get_default_hand_landmarks_style(), # type: ignore
        solutions.drawing_styles.get_default_hand_connections_style()) # type: ignore

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        avg_landmark_x[handedness[0].category_name] = np.average(x_coordinates) * width
        avg_landmark_y[handedness[0].category_name] = np.average(y_coordinates) * height
        all_handedness.add(handedness[0].category_name)

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
        # Get the thumb tip and index finger tip landmarks.
        if len(all_handedness) != 0:
            Thumb_Tip = [int(hand_landmarks[4].x * width), int(hand_landmarks[4].y * height)]
            Index_Finger_Tip = [int(hand_landmarks[8].x * width), int(hand_landmarks[8].y * height)]
        

    return annotated_image, avg_landmark_x, avg_landmark_y, all_handedness,Thumb_Tip,Index_Finger_Tip


def get_hand_speed(coordinates):
        '''
        Get the speed curve of the hand.
        '''
        length = len(coordinates)
        speed = np.zeros(length-1)
        for i in range(length-1):
            if coordinates[i] is not None and coordinates[i+1] is not None:
                speed[i] = np.linalg.norm(coordinates[i] - coordinates[i+1])
            else:
                speed[i] = None
        return speed



def get_hand_motion(img,all_landmark_pos,HandLandmarker,mp_hand_options,frame_counter):
    '''
    识别手部运动
    '''
    # convert the BGR image to RGB because of OpenCV uses BGR while MediaPipe uses RGB
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with HandLandmarker.create_from_options(mp_hand_options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) # convert the image to MediaPipe format
        mp_timestamp = int(round(time.time()*1000)) # get the current timestamp in milliseconds

        results = landmarker.detect_for_video(mp_image, mp_timestamp) # detect the hands in the frame

        # Draw the landmarks on the image and get the average x and y coordinates of the hand landmarks in this frame.
        annotated_image, avg_landmark_x, avg_landmark_y, all_handedness, Thumb, Index_finger = process_frame_results(rgb_image=img, 
                                                                                                detection_result=results)

        if len(all_handedness) == 0:
            # no hands detected in this frame
            all_landmark_pos['Right'][frame_counter] = None
            all_landmark_pos['Left'][frame_counter] = None
        else:
            for handedness in all_handedness:
                # collet both left and right hand landmarks
                all_landmark_pos[handedness][frame_counter] = (avg_landmark_x[handedness], avg_landmark_y[handedness])

    # save the image to a folder, not as a video
    # cv2.imwrite(f'{SAVE_TRACKING_RESULTS_DIR}/{frame_counter}.jpg', annotated_image)
    
    # annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return annotated_image, all_landmark_pos, Thumb, Index_finger