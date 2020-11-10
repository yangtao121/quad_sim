import numpy as np


# 将欧拉角转换成R矩阵
def Rotation_matrix(angle):
    roll = angle[0]
    pitch = angle[1]
    yaw = angle[2]

    cr = np.cos(roll)
    sr = np.sin(roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)

    cy = np.cos(yaw)
    sy = np.sin(yaw)

    R = np.array([
        [cr * cy - cp * sr * sy, -cy * sr - cr * cp * sy, sp * sy],
        [cp * cy * sr + cr * sy, cr * cp * cy - sr * sy, -cy * sp],
        [sr * sp, cr * sp, cp]
    ])

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
            angle[i] += 2*np.pi
        if angle[i] > 2*np.pi:
            angle[i] -= 2*np.pi

    return angle