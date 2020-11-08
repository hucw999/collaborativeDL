# linUCB算法
# dataset.txt数据集里面：第一列为推荐系统推荐的文章号，第二列为回报，第三列及之后都为文章特征
import numpy as np
import matplotlib.pyplot as plt


class ModelSelector():



    # 初始化
    def __init__(self, numFeatures, numArms):
        # 初始化A，b，θ，p
        self.A = np.array([np.eye(numFeatures).tolist()]*numArms)

        self.b = np.zeros((numArms, numFeatures,1))

        self.theta = np.zeros((numArms, numFeatures,1))
        self.p = np.zeros(numArms)
        self.numArms = numArms
# 训练
    def train(self, data_array):
        # CTR指数（点击率）
        ctr_list = []
        # 时间
        T = []

        num = 3

        # 可选的臂（根据数据）

        # 历史数据数
        trials = data_array.shape[0]
        # 臂特征数（根据数据）
        numFeatures = data_array.shape[1] - 2
        print(numFeatures)
        # 总回报
        total_payoff = 0
        count = 0

        mz = 0

        alpha_list = [0.01, 0.1, 0.5, 0.99]


        # 初始化
        # A, b, theta, p = init(numFeatures, numArms)
        for iter in range(num):
            for t in range(0, trials):
                # 每行数据
                row = data_array[t]
                # 第几个臂（数据第一列）
                arm = row[0] - 1
                # 回报（数据第二列）
                payoff = row[1]
                # 特征（数据第三列及之后）
                x_t = np.expand_dims(row[2:], axis=1)

                # alpha为探索程度，几种情况
                # Best Setting for Strategy 1
                alpha = 1-(np.sqrt(t)/float(t*t))

                # Best Setting for Strategy 2
                # i = 0.05
                # alpha = float(i) / np.sqrt(t + 1)

                # Best Setting for Strategy 3. Another best setting is i = 0.4
                # i = 0.1
                # alpha = float(i) / (t + 1)

                # 求每个臂的p
                for a in range(0, self.numArms):
                    # 求逆
                    A_inv = np.linalg.inv(self.A[a])

                    # 相乘
                    self.theta[a] = np.matmul(A_inv, self.b[a])


                    # 求臂的p
                    self.p[a] = np.matmul(self.theta[a].T, x_t) + alpha * np.sqrt(np.matmul(np.matmul(x_t.T, A_inv), x_t))



                # 更新Aat，bat
                self.A[arm] = self.A[arm] + np.matmul(x_t, x_t.T)
                self.b[arm] = self.b[arm] + payoff * x_t

                # print(b[arm])

                # 选择p最大的那个臂作为推荐
                best_predicted_arm = np.argmax(self.p)
                # 推荐的臂与所给的相同，将参与CTR点击率计算
                if best_predicted_arm == arm:
                    mz = mz + 1


                    total_payoff = total_payoff + payoff
                    count = count + 1
                    ctr_list.append(total_payoff / count)
                    T.append(iter * trials + t + 1)
                # count = count + 1
                # ctr_list.append(mz / count)
                # T.append(count)

        # print(T)
        # CTR趋势画图
        plt.xlabel("T")
        plt.ylabel("CTR")
        plt.plot(T, ctr_list)
        # 存入路径
        # plt.savefig('../data/ctrVsT/LinUCB.png')
        plt.show()

    def modelSelect(self, x_t):

        alpha = 0.1

        for a in range(0, self.numArms):
            # 求逆
            print(self.A.shape)
            A_inv = np.linalg.inv(self.A[a])

            # 相乘
            self.theta[a] = np.matmul(A_inv, self.b[a])

            # 求臂的p
            self.p[a] = np.matmul(self.theta[a].T, x_t) + alpha * np.sqrt(np.matmul(np.matmul(x_t.T, A_inv), x_t))

        # 更新Aat，bat
        # self.A[arm] = self.A[arm] + np.matmul(x_t, x_t.T)
        # b[arm] = b[arm] + payoff * x_t

        # print(b[arm])

        # 选择p最大的那个臂作为推荐
        best_predicted_arm = np.argmax(self.p)
        return best_predicted_arm

if __name__ == "__main__":
    # # 获取数据
    # data_array = np.loadtxt('../data/dataset.txt', dtype=int)
    # 训练
    # train(data_array)

    modelSelector = ModelSelector(3, 3)

    x_t = np.array([100, 3, 2])

    model = modelSelector.modelSelect(x_t)
    print(model)