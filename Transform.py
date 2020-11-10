import numpy as np


# 各个轴的旋转矩阵
def rotate_x(angel):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(angel), -np.sin(angel)],
        [0, np.sin(angel), np.cos(angel)]
    ])

    return R_x


def rotate_y(angel):
    R_y = np.array([
        [np.cos(angel), 0, np.sin(angel)],
        [0, 1, 0],
        [-np.sin(angel), 0, np.cos(angel)]
    ])
    return R_y


def rotate_z(angel):
    R_z = np.array([
        [np.cos(angel), -np.sin(angel), 0],
        [np.sin(angel), np.cos(angel), 0],
        [0, 0, 1]
    ])
    return R_z


# _________________________________________________


# 获得欧拉角矩阵，R_z*R_y*R_x
def Rotation_matrix(angle):
    """

    :param angle[x,y,z]
    :return:
    """
    R = np.dot(rotate_z(angle[2]),rotate_y(angle[1]),rotate_x(angle[0]))
    return R


def theta_dot2omega(theta_dot, angle):
    roll = angle[0]
    pitch = angle[1]
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)

    tr_matrix = np.array([
        [1, 0, -sp],
        [0, cr, cp * sr],
        [0, -sr, cp * cr]
    ])

    omega = np.dot(tr_matrix, theta_dot)
    return omega


def omega2theta_dot(omega, angle):
    roll = angle[0]
    pitch = angle[1]
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    tp = np.tan(pitch)

    tr_matrix = np.array([
        [1, sr * tp, cr * tp],
        [0, cr, -sr],
        [0, sr / cp, cr / cp]
    ])

    theta_dot = np.dot(tr_matrix, omega)
    return theta_dot


# 将角度圆整
def normalize_angle(angle):
    for i in range(3):
        if angle[i] < 0:
            angle[i] += 2 * np.pi
        if angle[i] > 2 * np.pi:
            angle[i] -= 2 * np.pi

    return angle
