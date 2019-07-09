########################################################################
#
# Copyright (c) 2018, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Multi cameras sample showing how to open multiple ZED in one program
"""
import time

import cv2
import pyzed.sl as sl
import datetime

PATH = '/home/apg/Desktop/Recordings/'


def main():
    print("Running...")
    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    init.camera_linux_id = 0
    init.camera_fps = 10  # The framerate is lowered to avoid any USB3 bandwidth issues
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera 1...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print('Camera 1: ' + repr(status))
        exit()

    # init.camera_linux_id = 1  # selection of the ZED ID
    # cam2 = sl.Camera()
    # if not cam2.is_opened():
    #     print("Opening ZED Camera 2...")
    # status = cam2.open(init)
    # if status != sl.ERROR_CODE.SUCCESS:
    #     print('Camera 2: ' + repr(status))
    #     exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()
    # mat2 = sl.Mat()

    print_camera_information(cam)
    # print_camera_information(cam2)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_c1 = None
    out_c2 = None

    while True:
        # Grab an image, a RuntimeParameters object must be given to grab()
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:  # and cam2.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            cam.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
            # cam2.retrieve_image(mat2, sl.VIEW.VIEW_LEFT)
            if out_c1 is None:  # or out_c2 is None:
                print("Saving the recording at: {}".format(PATH + time_stamp + '_cam1.avi'))
                out_c1 = cv2.VideoWriter(PATH + time_stamp + '_cam1.avi', fourcc, 20.0,
                                         (int(mat.get_width()), int(mat.get_height())))
                # out_c2 = cv2.VideoWriter(time_stamp + '_cam2.avi', fourcc, 20.0, (int(mat2.get_width()),
                #                                                                   int(mat2.get_height())))

            c1_frame = mat.get_data()
            # c2_frame = mat2.get_data()

            # Save the frames as a video
            if out_c1 is not None:
                out_c1.write(c1_frame[:, :, :3])
            # out_c2.write(c2_frame)

            # Display the image frames
            cv2.imshow('Camera 1 - Left', c1_frame)
            # cv2.imshow('Camera 2 - Left', c2_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.close()
    # cam2.close()

    out_c1.release()
    # out_c2.release()

    cv2.destroyAllWindows()
    print("\nFINISH")


def print_camera_information(cam):
    print("Resolution: {0}, {1}.".format(
        round(cam.get_resolution().width, 2), cam.get_resolution().height))
    print("Camera FPS: {0}.".format(cam.get_camera_fps()))
    print("Firmware: {0}.".format(
        cam.get_camera_information().firmware_version))
    print("Serial number: {0}.\n".format(
        cam.get_camera_information().serial_number))


if __name__ == "__main__":
    main()
