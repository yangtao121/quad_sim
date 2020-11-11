import numpy as np
import Transform as tr


class simple_quad_model:
    def __init__(self, m, g, I, l, h=0.01):
        """

        :param m: 无人机质量
        :param g: 重力加速度
        :param I: 转动惯量，为字典{'Ix','Iy','Iz'}
        :param h: 仿真步长
        :param l: 翼到中心的距离
        """
        self.g = g
        self.m = m
        self.l = l
        self.h = h

        # self.B_angel = np.zeros(3)  # 相对于机体坐标系B的旋转角
        self.E_angel = np.zeros(3)  # 相对于绝对坐标系E的旋转角
        self.liner = np.zeros(3)  # 线位置
        self.I = I
        self.angel_speed = np.zeros(3)
        self.liner_speed = np.zeros(3)

        # self.input = np.zeros(4)  # 旋翼输入力

        # 与强化学习相关
        self.Time_counter = 0  # 时间计数器

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
        # print(U)
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
        # print(acc)
        return acc

    def angel_acceleration(self, U):
        acc = np.zeros(3)
        acc[0] = (self.angel_speed[1] * self.angel_speed[2] * (self.I['Iy'] - self.I['Iz']) + self.l * U[1]) / self.I[
            'Ix']
        acc[1] = (self.angel_speed[0] * self.angel_speed[2] * (self.I['Iz'] - self.I['Ix']) + self.l * U[2]) / self.I[
            'Iy']
        acc[2] = (self.angel_speed[0] * self.angel_speed[1] * (self.I['Ix'] - self.I['Iy']) + U[3]) / self.I['Iz']
        # print(acc)
        return acc

    def sim_speed(self, U):
        """
        使用最简单的仿真，V(t)=V0+at
        :param U: 虚拟输入量
        :return: 返回i+1时刻的速度值
        """
        acc_liner = self.liner_acceleration(U)
        acc_angel = self.angel_acceleration(U)
        liner_speed = acc_liner * self.h + self.liner_speed
        angel_speed = acc_angel * self.h + self.angel_speed

        return liner_speed, angel_speed

    def sim_state(self):
        """
        p=p0+Vt
        :return: liner, E_angel
        """
        liner = self.liner + self.liner_speed * self.h
        E_angel = self.E_angel + self.angel_speed * self.h
        E_angel = tr.normalize_angle(E_angel)

        return liner, E_angel

    def step(self, F):
        U = self.virtual_control_U(F)
        self.liner_speed, self.angel_speed = self.sim_speed(U)
        self.liner, self.E_angel = self.sim_state()
        # print(np.array([
        #     self.liner_speed,
        #     self.angel_speed,
        #     self.liner,
        #     self.E_angel
        # ]))

    # 强化学习相关的函数
    def reset(self):
        """
        获得一个不平衡状态
        :return:
        """
        self.E_angel = np.random.uniform(0, np.pi / 3, 3)
        self.angel_speed = np.random.uniform(-np.pi / 4, np.pi / 4, 3)
        self.liner_speed = np.random.uniform(-5, 5, 3)
        state = np.array([self.liner_speed,self.angel_speed,self.E_angel])
        state = state.flatten()
        return state

    def reward(self):
        """
        奖励值的定义：
        reward = -cost = -(k1*speed^2 + k2*angel^2)
        :return: 返回奖励值
        """
        reward = -(0.1 * (np.square(self.angel_speed).sum() + np.square(self.liner_speed).sum()) + 0.2 * np.square(
            self.E_angel).sum())
        print(reward)
        return reward

    def reinforce_step(self, F):
        """
        为强化学习专门定制的step
        :param F: 力的输入
        :return: 返回线速度+角速度+欧拉角
        """
        U = self.virtual_control_U(F)
        self.liner_speed, self.angel_speed = self.sim_speed(U)
        self.liner, self.E_angel = self.sim_state()
        reward = self.reward()
        self.Time_counter += 1
        # print(self.Time_counter)
        if self.Time_counter % 500 == 0:
            done = True
        else:
            done = False

        state = np.array([self.liner_speed, self.angel_speed, self.E_angel])
        state = state.flatten()

        return state, reward, done
