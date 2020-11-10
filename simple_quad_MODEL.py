import numpy as np


class simple_quad_model:
    def __init__(self, m, g, I):
        """

        :param m: 无人机质量
        :param g: 重力加速度
        :param I: 转动惯量，为字典{'Ix','Iy','Iz'}
        """
        self.g = g
        self.m = m

        # self.B_angel = np.zeros(3)  # 相对于机体坐标系B的旋转角
        self.E_angel = np.zeros(3)  # 相对于绝对坐标系E的旋转角
        self.E_angel[1] = np.pi/3
        self.I = I

        # self.input = np.zeros(4)  # 旋翼输入力

    def virtual_control_U(self, F):
        """
        返回值说明：
        U[0]垂直速度控制量
        U[1]横滚控制量
        U[2]俯仰控制量
        U[3]偏航控制量
        :param F: 输入的真实力
        :return:
        """
        U = np.zeros(4)
        U[0] = np.sum(F)
        U[1] = F[3] - F[1]
        U[2] = F[3] - F[0]
        U[3] = F[1] + F[3] - F[2] - F[0]
        return U

    def liner_acceleration(self, U):
        """

        :param U: 虚拟输入量
        :return: 线加速度，0，1，2：x,y,z
        """
        acc = np.zeros(3)
        # U = self.virtual_control_U(F)
        acc[0] = (np.cos(self.E_angel[0]) * np.sin(self.E_angel[1]) * np.cos(self.E_angel[2]) + np.sin(
            self.E_angel[0]) * np.sin(self.E_angel[2])) * U[0] / self.m
        acc[1] = (np.cos(self.E_angel[0]) * np.sin(self.E_angel[1]) * np.sin(self.E_angel[2]) - np.sin(
            self.E_angel[0]) * np.cos(self.E_angel[2])) * U[0] / self.m
        acc[2] = (np.cos(self.E_angel[1]) * np.cos(self.E_angel[0])) * U[0] / self.m - self.g
        return acc
