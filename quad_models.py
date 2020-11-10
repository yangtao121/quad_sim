import numpy as np
import Transform as Tr


class quad_models:
    """
    Ct为单桨综合拉力系数
    L为多旋翼机身半径
    Cm为单桨综合扭矩系数
    Cd为阻力系数
    I无人机惯性矩阵(3X3)
    m无人机的质量
    max_v最大飞行速度
    g重力加速度
    dt仿真精度
    """

    def __init__(self, Ct, L, Cm, Cd, I, m, max_v, g=10, dt=0.005):
        # 四旋翼无人机参数设置
        self.Ct = Ct  # 单桨综合拉力系数
        self.L = L  # 多旋翼机身半径
        self.Cm = Cm  # 单桨综合扭矩系数
        self.Cd = Cd  # 无人机的阻力系数
        self.I = I
        self.m = m

        self.Ix = I[0, 0]
        self.Iy = I[1, 1]
        self.Iz = I[2, 2]

        self.max_v = max_v

        # 状态参数
        self.omega_rpm = np.array([0, 0, 0, 0])  # 螺旋桨转速,单位rad/s
        self.omega = np.array([0, 0, 0])  # 相对于自身坐标系的角速度
        self.Euler_angle = np.array([0, 0, 0])  # 或者看作theta角
        self.x = np.array([0, 0, 0])  # 无人机在空间的位置
        self.x_dot = np.array([0, 0, 0])  # 无人机的速度
        self.theta_dot = np.array([0, 0, 0])

        # 环境参数、仿真参数设置
        self.g = g
        self.dt = dt

        # 学习参数的设定
        self.w1 = 0.1
        self.w2 = 0.1
        self.w3 = 0.01
        self.step_1 = 0

    def thrust(self):
        square_omega = self.omega_rpm ** 2
        # 总推力
        TB = self.Ct * np.array([0, 0, np.sum(square_omega)])
        return TB

    # 计算机身力矩
    def tau(self):
        tao_x = self.L * self.Ct * (self.omega_rpm[0] ** 2 - self.omega_rpm[2] ** 2)
        tao_y = self.L * self.Ct * (self.omega_rpm[1] ** 2 - self.omega_rpm[3] ** 2)

        # 计算tao_z
        flag = np.array([1, -1, 1, -1])
        tao_z = self.Cm * (np.sum(flag * self.omega_rpm ** 2))

        return np.array([tao_x, tao_y, tao_z])

    # 计算线性加速度
    def liner_acceleration(self):
        Rotation_matrix = Tr.Rotation_matrix(self.Euler_angle)
        gravity = np.array([0, 0, -self.g])
        # 无人机的线速度阻力
        Fd = -self.Cd * self.x_dot ** 2

        # 在绝对坐标系下的提升力
        T = np.dot(Rotation_matrix, self.thrust())

        a = gravity + T + Fd
        return a

    # 计算角角加速度,相对于无人机自身坐标系
    def angel_acceleration(self):
        a1 = np.dot(np.linalg.inv(self.I), self.tau())

        a2_x = (self.Iy - self.Iz) / self.Ix * self.omega[1] * self.omega[2]
        a2_y = (self.Iz - self.Ix) / self.Iy * self.omega[0] * self.omega[2]
        a2_z = (self.Ix - self.Iy) / self.Iz * self.omega[0] * self.omega[1]

        omega_dot = a1 - np.array([a2_x, a2_y, a2_z])
        return omega_dot

    # 改变螺旋桨转速，这里需要设置电机转速响应
    def change_rpm(self, action):
        self.omega_rpm = action

    def get_reward(self, x_dot, theta_dot, Euler):
        if np.sum(x_dot ** 2) < 400:
            reward = -(self.w1 * np.sum(x_dot ** 2))# + self.w2 * np.sum(theta_dot ** 2) + self.w3 * np.sum(Euler ** 2))
        if self.step_1 >= 1499 or np.sum(x_dot ** 2) > 400:
            reward = -1e7
        return reward

    # 判断是否结束
    def judge_done(self, x_dot, theta_dot, Euler):
        a = np.array([np.sum(np.abs(x_dot))]) #, np.sum(np.abs(theta_dot)), Euler[0], Euler[1]])
        # print(x_dot)
        # print(theta_dot)
        # print(Euler)
        # print("----------------------------------------")
        if np.sum(np.abs(a)) < 1 or np.sum(x_dot ** 2) > 400:
            return True
        else:
            return False

    def reset(self):
        self.x_dot = np.random.uniform(-10, 10, 3)
        self.omega_rpm = np.array([486.13, 486.13, 486.13, 486.13])
        self.omega = np.random.uniform(-2, 2, 3)
        self.step_1 = 0
        state = np.array([self.x_dot, self.omega, self.Euler_angle])
        return state.flatten()

    def step(self, action):
        self.change_rpm(action)

        reward = self.get_reward(self.x_dot, self.theta_dot, self.Euler_angle)

        done = self.judge_done(self.x_dot, self.theta_dot, self.Euler_angle)

        # 简单的仿真，分线速度和角速度
        a = self.liner_acceleration()
        omega_dot = self.angel_acceleration()
        self.omega = self.omega + omega_dot * self.dt
        self.omega = Tr.normalize_angle(self.omega)
        self.theta_dot = Tr.omega2theta_dot(self.omega, self.Euler_angle)
        self.Euler_angle = self.Euler_angle + self.theta_dot * self.dt
        self.Euler_angle = Tr.normalize_angle(self.Euler_angle)

        self.x_dot = self.x_dot + a * self.dt
        self.x = self.x + self.x_dot * self.dt
        state = np.array([self.x_dot, self.omega, self.Euler_angle])
        self.step_1 += 1

        return state.flatten(), reward, done
