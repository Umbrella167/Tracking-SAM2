import numpy as np

class KalmanFilter:
    def __init__(self, state_dim, measure_dim, dt=1.0):
        """
        初始化卡尔曼滤波器
        :param state_dim: 状态向量的维度
        :param measure_dim: 测量向量的维度
        :param dt: 时间间隔 (默认值为 1.0)
        """
        self.state_dim = state_dim
        self.measure_dim = measure_dim
        self.dt = dt

        # 状态向量 (初始状态)
        self.x = np.zeros((state_dim, 1))

        # 状态转移矩阵 (F)
        self.F = np.eye(state_dim)
        
        # 控制输入矩阵(B, 可选)
        self.B = None
        
        # 测量矩阵 (H)
        self.H = np.zeros((measure_dim, state_dim))

        # 状态协方差矩阵 (P)
        self.P = np.eye(state_dim)

        # 过程噪声协方差矩阵 (Q)
        self.Q = np.eye(state_dim)

        # 测量噪声协方差矩阵 (R)
        self.R = np.eye(measure_dim)

    def predict(self, u=None):
        """
        预测步骤
        :param u: 控制输入向量 (如果有控制输入)
        :return: 预测后的状态向量
        """
        if u is not None and self.B is not None:
            self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        else:
            self.x = np.dot(self.F, self.x)
        
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """
        更新步骤
        :param z: 测量向量
        :return: 更新后的状态向量
        """
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 更新状态向量
        y = z - np.dot(self.H, self.x)  # 预测残差
        self.x = self.x + np.dot(K, y)

        # 更新状态协方差矩阵
        I = np.eye(self.state_dim)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x

    def set_state(self, x, P=None):
        """
        设置初始状态
        :param x: 初始状态向量
        :param P: 初始协方差矩阵 (可选)
        """
        self.x = np.array(x).reshape(-1, 1)
        if P is not None:
            self.P = P

    def configure_matrices(self, F=None, H=None, Q=None, R=None, B=None):
        """
        配置矩阵
        :param F: 状态转移矩阵
        :param H: 测量矩阵
        :param Q: 过程噪声协方差矩阵
        :param R: 测量噪声协方差矩阵
        :param B: 控制输入矩阵
        """
        if F is not None:
            self.F = F
        if H is not None:
            self.H = H
        if Q is not None:
            self.Q = Q
        if R is not None:
            self.R = R
        if B is not None:
            self.B = B