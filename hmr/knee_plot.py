#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

data_hip = pd.read_csv('/home/apg/Desktop/Recordings/left_hip_2019-05-22-13:23:46.csv', delimiter=',', header=None)

z = np.linspace(0, 535, 10480)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot3D(data_hip[0], data_hip[1], z, 'gray')
# plt.title('Left Hip')
# plt.show()

###############################################


data_knee = pd.read_csv('/home/apg/Desktop/Recordings/left_knee_2019-05-22-13:23:46.csv', delimiter=',', header=None)

#
# final = []
#
# for index in range(0, len(data_knee[0])):
#     final.append((z[index], data_knee[1][index]))
#
# plt.figure(1)
# plt.plot(*zip(*final))
# plt.title('Joint anlge, Left Knee')
# plt.show()


data_ankle = pd.read_csv('/home/apg/Desktop/Recordings/left_ankle_2019-05-22-13:23:46.csv', delimiter=',', header=None)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot3D(data_ankle[0], data_ankle[1], z, 'gray')
# plt.title('Left Ankle')
# plt.show()


angles = []


for index in range(0, len(data_hip[0])):
    a1 = (data_knee[1][index] - data_hip[1][index])/(data_knee[0][index] - data_hip[0][index])
    a2 = (data_knee[1][index] - data_ankle[1][index])/(data_knee[0][index] - data_ankle[0][index])

    import math
    temp = (a2-a1)/(1+a2*a1)
    # if temp < 0:
    #     print(math.atan(temp))
    #     temp = math.pi - math.atan(temp)
    # else:
    #     temp = math.atan(temp)
    angles.append(math.atan(temp))
#
# # fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.plot(angles, z)
# # plt.title('Joint angle')
# # plt.show()

print(len(angles))
print(len(z))

final = []

for index in range(0, len(angles), 2):
    final.append((z[index], angles[index]))

plt.figure(1)
plt.plot(*zip(*final))
plt.title('Joint anlge, Left Knee')
plt.show()