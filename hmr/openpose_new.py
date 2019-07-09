# From Python
# It requires OpenCV installed for Python
import datetime
import sys
import cv2
import os
from sys import platform
import argparse
import time

try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../../../examples/media/",
                    help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/home/apg/openpose/models/"
params["write_json"] = "/home/apg/Desktop/hmr/keypoint/"

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

try:
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    PATH = '/home/apg/Desktop/Recordings/'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = '2019-04-05-16:46:59_cam1_biking_new.avi'
    list_name = 'bike_data_' + str(time_stamp) + '.csv'
    cap = cv2.VideoCapture(PATH + video_name)
    #print("FPS: {}".format(cap.get(cv2.CAP_PROP_FPS)))

    start = time.time()
    ret, frame = cap.read()
    print(ret)
    while ret:
        ret, frame = cap.read()
        print(ret)
        if frame is not None:
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            # print("Body keypoints: \n" + str(datum.poseKeypoints))

            # if not args[0].no_display:
            #     cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
            #     key = cv2.waitKey(15)
            #     if key == 27: break
        cv2.waitKey(15)

    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")

    # myfile.close()
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    print("Complete.")


except Exception as e:
    # print(e)
    sys.exit(-1)
