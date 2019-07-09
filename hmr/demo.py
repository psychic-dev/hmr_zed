"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import datetime
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import cv2
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    # plt.ion()
    # plt.figure(1)
    # plt.clf()
    # plt.subplot(231)
    # plt.imshow(img)
    # plt.title('input')
    # plt.axis('off')
    # plt.subplot(232)
    # plt.imshow(skel_img)
    out.write(skel_img)

    # plt.title('joint projection')
    # plt.axis('off')
    # plt.subplot(233)
    # plt.imshow(rend_img_overlay)
    # plt.title('3D Mesh overlay')
    # plt.axis('off')
    # plt.subplot(234)
    # plt.imshow(rend_img)
    # plt.title('3D mesh')
    # plt.axis('off')
    # plt.subplot(235)
    # plt.imshow(rend_img_vp1)
    # plt.title('diff vp')
    # plt.axis('off')
    # plt.subplot(236)
    # plt.imshow(rend_img_vp2)
    # plt.title('diff vp')
    # plt.axis('off')
    # plt.draw()
    # plt.pause(0.001)
    # plt.show()
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img, id, json_path=None):
    # img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path, id)

    if scale is not None and center is not None:
        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                                       config.img_size)

        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)

        return crop, proc_param, img
    else:
        return None, None, None


def plot_knee_left(tuple):
    # plt.clf()
    plt.plot(tuple[0], tuple[1], 'r')
    plt.draw()
    plt.pause(0.001)


def main(img_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    # plt.figure(1)
    ret, frame = cap.read()
    id = 0
    bad_frames_count = 0
    # left_knee_list = []
    # left_hip_list = []
    # left_ankle_list = []
    while ret:
        ret, frame = cap.read()

        # uncomment to use openpose keypoints
        if frame is not None:
            # input_img, proc_param, img = preprocess_image(frame, id, json_path + str(id) + '_keypoints.json')
            input_img, proc_param, img = preprocess_image(frame, id, None)

            # Add batch dimension: 1 x D x D x 3
            if input_img is not None:
                input_img = np.expand_dims(input_img, 0)

                # Theta is the 85D vector holding [camera, pose, shape]
                # where camera is 3D [s, tx, ty]
                # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
                # shape is 10D shape coefficients of SMPL
                joints, verts, cams, joints3d, theta = model.predict(input_img, get_theta=True)
                # pose = theta[0][2:-11]
                #
                #
                #
                # cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
                #     proc_param, verts, cams[0], joints, img_size=img.shape[:2])
                #
                # # j = joints_orig[0]
                # j = joints3d[0]
                #
                #
                # left_hip_index = 3
                # left_hip = j[left_hip_index]
                # print(left_hip)
                # # left_hip = joints3d[3]
                # # left_hip_list.append(left_hip)
                #
                # left_knee_index = 4
                # left_knee = j[left_knee_index]
                # # left_knee_list.append(left_knee)
                #
                # left_ankle_index = 5
                # left_ankle = j[left_ankle_index]
                # # left_ankle_list.append(left_ankle)
                #
                #
                # wr.writerow(left_hip)
                # wr1.writerow(left_knee)
                # wr2.writerow(left_ankle)

                visualize(img, proc_param, joints[0], verts[0], cams[0])
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                bad_frames_count += 1
                print('Bad person frame# :{}'.format(id))

        id += 1
        print("-----------------------------------------------------------")


    # Data for a three-dimensional line
    # zline = np.linspace(0, 535, 10700)
    # xline = [elem1 for elem1, elem2 in left_knee_list]
    # yline = [elem2 for elem1, elem2 in left_knee_list]
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot3D(xline, yline, zline, 'gray')

    # plt.subplot(211)
    # plt.plot(*zip(*left2))
    # plt.title('Left Knee')
    #
    # right2 = [(elem1, elem2) for elem1, elem2 in right_knee_list]
    # plt.subplot(212)
    # plt.plot(*zip(*right2))
    # plt.title('Right Knee')
    # plt.show()

    print("Total bad frames: {}".format(bad_frames_count))
    cap.release()
    out.release()
    # myfile.close()
    # myfile1.close()
    # myfile2.close()

if __name__ == '__main__':
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    PATH = '/home/apg/Desktop/Recordings/Videos/'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_name = 'biking_new.avi'
    list_name = 'left_hip_' + str(time_stamp) + '.csv'
    list_name1 = 'left_knee_' + str(time_stamp) + '.csv'
    list_name2 = 'left_ankle_' + str(time_stamp) + '.csv'
    cap = cv2.VideoCapture(PATH + video_name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
    out = cv2.VideoWriter(PATH + 'output.avi', fourcc, 20.0, (width, height))

    print('Video FPS: {}'.format(cap.get(cv2.CAP_PROP_FPS)))

    # myfile = open(PATH + list_name, 'wb')
    # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #
    # myfile1 = open(PATH + list_name1, 'wb')
    # wr1 = csv.writer(myfile1, quoting=csv.QUOTE_ALL)
    #
    # myfile2 = open(PATH + list_name2, 'wb')
    # wr2 = csv.writer(myfile2, quoting=csv.QUOTE_ALL)

    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, '/home/apg/Desktop/hmr/keypoint/')
