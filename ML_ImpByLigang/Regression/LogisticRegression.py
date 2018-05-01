'''
LogisticRegression 回归的试例
西瓜书 3.3  手动实现 对率回归
对率回归实际上解决的是 分类问题

'''
import numpy as np
def LogisticFunction(x, beta):
    z = np.dot(x, beta)
    g = 1 / ( 1 + np.exp(-z))
    return g

def LogisticRegressionWithNewton(x, beta, y):
#     利用极大似然法求解Logistics回归的最优解， 利用牛顿法进行具体的求解过程
#       x1 of x  equals  (x;1)  and  beta equals (w;b)
#       求一阶导、二阶导  然后更新值

    for i in range(0,15):
        p1 = 1 -  LogisticFunction(x, beta)
        betaFirstDerivative = np.dot(x.T, (y-p1).reshape(y.shape[0],1))
        xSecondNormalForm  = np.dot(x.T, x)
        betaSecondDerivative = xSecondNormalForm * (p1*(1-p1)).sum()
        betaOld = beta
        beta = beta - np.dot(np.asarray(np.mat(betaSecondDerivative).I), betaFirstDerivative)
        print("betaOld \t beta:\n", betaOld, beta)


def  LogisticRegressionWithNewton_Forloop(x, beta, y):
    n = x.shape[1]
    m = x.shape[0]
    G = np.zeros((n, 1))
    H = np.zeros((n, n))

    for interation in range(0,10):
        h = LogisticFunction(x, beta)
        diff = y - h
        betaOld = beta
        for i in range(0, n):
            G[i] = np.sum(diff.T*x[:, i].T) / m
            const_sum = (h.T*(1-h.T))
            for j in range (0, n):
                H[i,j]  = np.sum(const_sum * x[:, i] * x[:, j]) / m
        beta = beta -  np.dot(np.asarray(np.mat(H).I), G)
        print(beta.shape)
        print("betaOld \t beta:\n", betaOld, beta)


if __name__ ==  "__main__":
    # 初始化西瓜数据集
    dataSet = np.array([[0.697, 0.460, 1],
                         [0.774, 0.376, 1],
                         [0.608, 0.318, 1],
                         [0.556, 0.215, 1],
                         [0.403, 0.237, 1],
                         [0.481, 0.149, 1],
                         [0.437, 0.211, 1],
                         [0.666, 0.091, 0],
                         [0.243, 0.267, 0],
                         [0.245, 0.267, 0],
                         [0.343, 0.099, 0],
                         [0.639, 0.161, 0],
                         [0.657, 0.198, 0],
                         [0.360, 0.370, 0],
                         [0.593, 0.042, 0],
                         [0.719, 0.103, 0]])
    print(dataSet[0])
    y = dataSet[:, 2].reshape(16,1)
    x = dataSet[:, 0:2]
    x = np.insert(x, 2, np.ones(x.shape[0]), 1)
    beta = np.array([0,0,0]).reshape(3,1)
    LogisticRegressionWithNewton(x, beta, y)
    # LogisticRegressionWithNewton_Forloop(x, beta, y)





