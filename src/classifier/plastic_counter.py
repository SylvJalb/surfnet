import json
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import torch as th 
import cv2

sys.path.append('../')

from eval import * 
from collections import defaultdict
from tools.video_readers import SimpleVideoReader
from track import *
from detection.detect import detect
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout,SpatialDropout2D
from keras.models  import Sequential

# All parameters
device = torch.device("cpu")
arch = 'mobilenet_v3_small'
model_weights = None
detection_threshold = 0.3
noise_covariances_path = '../../data/tracking_parameters'
model_path = './models/trained_classifier.h5'
skip_frames = 3
output_shape = (960,544)
preload_frames = False
downsampling_factor = 1
detection_batch_size = 1
confidence_threshold = 0.2
display = Display(on=False, interactive=True)
crop_size = 100
skip_frames = 3


def associate_detections_to_trackers2(detections_for_frame, trackers, flow01):
    tracker_nbs, confidence_functions = build_confidence_function_for_trackers(trackers, flow01)
    assigned_trackers = [None]*len(detections_for_frame)
    if len(tracker_nbs):
        cost_matrix = np.zeros(shape=(len(detections_for_frame),len(tracker_nbs)))
        for detection_nb, detection in enumerate(detections_for_frame):
            for tracker_id, confidence_function in enumerate(confidence_functions):
                score = confidence_function(detection)
                if score > confidence_threshold:
                    cost_matrix[detection_nb,tracker_id] = score
                else:
                    cost_matrix[detection_nb,tracker_id] = 0
        row_inds, col_inds = linear_sum_assignment(cost_matrix,maximize=True)
        for row_ind, col_ind in zip(row_inds, col_inds):
            if cost_matrix[row_ind,col_ind] > confidence_threshold: assigned_trackers[row_ind] = tracker_nbs[col_ind]
        
    return assigned_trackers
    
def track_video2(reader, detections, downsampling_factor, engine, transition_variance, observation_variance):
    init = False
    trackers = dict()
    frame_nb = 0
    frame0 = next(reader)
    detections_for_frame = next(detections)

    max_distance = euclidean(reader.output_shape, np.array([0,0]))
    delta = 0.05*max_distance

    if display.on: 
    
        display.display_shape = (reader.output_shape[0] // downsampling_factor, reader.output_shape[1] // downsampling_factor)
        display.update_detections_and_frame(detections_for_frame, frame0)

    if len(detections_for_frame):
        trackers = init_trackers(engine, detections_for_frame, frame_nb, transition_variance, observation_variance, delta)
        init = True

    if display.on: display.display(trackers)

    for frame_nb, (frame1, detections_for_frame) in enumerate(zip(reader, detections), start=1):

        if display.on: display.update_detections_and_frame(detections_for_frame, frame1)

        if not init:
            if len(detections_for_frame):
                trackers = init_trackers(engine, detections_for_frame, frame_nb, transition_variance, observation_variance, delta)
                init = True

        else:

            new_trackers = []
            flow01 = compute_flow(frame0, frame1, downsampling_factor)

            if len(detections_for_frame):

                assigned_trackers = associate_detections_to_trackers2(detections_for_frame, trackers, flow01)

                for detection, assigned_tracker in zip(detections_for_frame, assigned_trackers):
                    if in_frame(detection, flow01.shape[:-1]):
                        if assigned_tracker is None :
                            new_trackers.append(engine(frame_nb, detection, transition_variance, observation_variance, delta))
                        else:
                            trackers[assigned_tracker].update(detection, flow01, frame_nb)

            for tracker in trackers:
                tracker.update_status(flow01)

            if len(new_trackers):
                trackers.extend(new_trackers)

        if display.on: display.display(trackers)
        frame0 = frame1.copy()


    results = []
    tracklets = [tracker.tracklet for tracker in trackers]
    
    for tracker_nb, associated_detections in enumerate(tracklets):
        for associated_detection in associated_detections:
            results.append((associated_detection[0], tracker_nb, associated_detection[1][0], associated_detection[1][1]))

    results = sorted(results, key=lambda x: x[0])
 
    return results

def get_cropped_images(input_video, input_mot_file, crop_size, skip_frames):
    # get the frame corresponding to the input_mot_file line
    ret = True 
    crop_images = []

    # Get all the first frame of an object
    nb_object = 0
    first_appear = []
    for line in input_mot_file:
        if ((nb_object+1) == int(line[1])):
            first_appear.append([int(line[0]) - 1, int(line[1]), int(float(line[2])), int(float(line[3]))])
            nb_object += 1

    current_frame_number = 0
    cpt = 0

    video = SimpleVideoReader(input_video, skip_frames=skip_frames)

    img_base = [[[0, 0, 0] for i in range(crop_size * 2)] for j in range(crop_size * 2)]
    img_base = np.array(img_base)

    while (ret and (cpt < nb_object)):
        
        ret, frame, frame_nb = video.read()
        # detections_for_frame = results[frame_nb]

        while ((cpt < len(first_appear)) and ((current_frame_number) == first_appear[cpt][1])):
            # crop the image

            crop_img = img_base.copy()
            
            for i in range(0, (crop_size * 2)):
            # for i in range(max((first_appear[cpt][2] - crop_size), 0), min((first_appear[cpt][2] + crop_size), len(frame))):
                for j in range(0, (crop_size * 2)):
                # for j in range(max((first_appear[cpt][3] - crop_size), 0), min((first_appear[cpt][3] + crop_size), len(frame))):
                    if ((first_appear[cpt][2] - crop_size + i > 0) and (first_appear[cpt][2] + crop_size + i < len(frame)) 
                    and (first_appear[cpt][3] - crop_size + j > 0) and (first_appear[cpt][3] + crop_size + j < len(frame[i]))):
                        crop_img[i][j] = frame[first_appear[cpt][2] - crop_size + i][first_appear[cpt][3] - crop_size + j]

            #cv2.imwrite(crop_folder + 'crop' + str(int(cpt)) + '.jpg', crop_img) 
            crop_images.append(crop_img)

            cpt += 1
        
        current_frame_number += 1
    return crop_images

def countTrashes(video_path):

    ############# Detect trashes #############

    engine = get_tracker('EKF')

    print('---Loading model...')            
    model = load_model(arch=arch, model_weights=model_weights, device=device)
    print('Model loaded.')

    detector = lambda frame: detect(frame, threshold=detection_threshold, model=model)

    transition_variance = np.load(os.path.join(noise_covariances_path, 'transition_variance.npy'))
    observation_variance = np.load(os.path.join(noise_covariances_path, 'observation_variance.npy'))

    print(f'---Processing {video_path}')        
    reader = IterableFrameReader(video_filename=os.path.join(video_path), 
                                    skip_frames=skip_frames,
                                    output_shape=(960,544),
                                    progress_bar=True,
                                    preload=preload_frames)


    input_shape = reader.input_shape
    output_shape = reader.output_shape
    ratio_y = input_shape[0] / (output_shape[0] // downsampling_factor)
    ratio_x = input_shape[1] / (output_shape[1] // downsampling_factor)

    print('Detecting...')
    detections = get_detections_for_video(reader, detector, batch_size=detection_batch_size, device=device)

    print('Tracking...')
    results = track_video2(reader, iter(detections), downsampling_factor, engine, transition_variance, observation_variance)

    ############# Get croped images #############
    crop_images = get_cropped_images(video_path, results, crop_size, skip_frames)

    ############# Detect classes #############
    classes = {0: 'Bottle', 1: 'Bottle_cap', 2: 'Broken_glass', 3: 'Can', 4: 'Carton', 5: 'Cigarette', 6: 'Cup', 7: 'Other_plastic', 8: 'Others', 9: 'Paper', 10: 'Plastic_bag_and_wrapper', 11: 'Straw', 12: 'Styrofoam_piece'}
    class_count = {'Bottle': 0, 'Bottle_cap': 0, 'Broken_glass': 0, 'Can': 0, 'Carton': 0, 'Cigarette': 0, 'Cup': 0, 'Other_plastic': 0, 'Others': 0, 'Paper': 0, 'Plastic_bag_and_wrapper': 0, 'Straw': 0, 'Styrofoam_piece': 0}
    for crop_image in crop_images:
        # preprocess the image
        crop_image = cv2.resize(crop_image, (200,200))
        # load model and predict
        model = keras.models.load_model(model_path)
        prediction = model.predict(crop_image.reshape(1,200,200,3))
        class_count[classes[np.argmax(prediction)]] += 1
    
    return class_count


print(countTrashes('../../data/validation_videos/T1/T1_1080_px_converted.mp4'))