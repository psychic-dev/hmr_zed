#!/usr/bin/env python
import csv
import datetime
import os
import sys
import time
import cv2
import numpy as np

sys.path.append('/usr/local/python')
from openpose import *

#
params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1
params["render_threshold"] = 0.05
params["tracking"] = 5
params["number_people_max"] = 1
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = '/home/apg/openpose/models/'
# Construct OpenPose object allocates GPU memory
openpose = OpenPose(params)

if __name__ == '__main__':
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    PATH = '/home/apg/Desktop/Recordings/'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = '2019-04-05-16:46:59_cam1_biking.avi'
    list_name = 'bike_data_' + str(time_stamp) + '.csv'
    cap = cv2.VideoCapture(PATH + video_name)
    data = []
    # myfile = open(PATH + list_name, 'wb')
    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    out = None


    while cap.isOpened():
        ret, frame = cap.read()
        start = time.time()
        rect = None
        if out is None:
            out = cv2.VideoWriter(PATH + '_openpose.avi', fourcc, 20.0,
                                (int(frame.shape[1]), int(frame.shape[0])))

        keypoints, output_image = openpose.forward(frame, True)
        if len(keypoints) > 0:
            b_parts = keypoints[0]  # consider only 1 person detection
            pts = []
            for i in range(len(b_parts)):
                # print 'x: {}, y:{}, confidence: {}'.format(b_parts[i][0], b_parts[i][1], b_parts[i][2])

                if 8 <= i <= 14 or 19 <= i <= 24:
                    pts.append((b_parts[i][0], b_parts[i][1]))
            # wr.writerow(pts)

        # Compute the time for this loop and estimate CPS as a running average
        end = time.time()
        duration = end - start
        fps = int(1.0 / duration)

        cv2.putText(frame, "FPS: " + str(fps), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0))


        # if rect is not None:
        #     cv2.rectangle(output_image, (rect[0], rect[1]),
        #                   (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)

        # cv2.imshow('frame', output_image)
        out.write(output_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # myfile.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print 'Complete'
