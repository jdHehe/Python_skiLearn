'''
LDA 线性判别分析
通过将原数据集投影到新的区间（直线、平面等），实现同类聚集、异类分散的目的，同时可以降维
LDA区别与PCA主成分分析
'''

import  pandas as  pd
from sklearn.preprocessing import LabelEncoder
import numpy as np



def LoadData():
    feature_dict = {i: label for i, label in zip(
        range(4),
        ('sepal length in cm',
         'sepal width in cm',
         'petal length in cm',
         'petal width in cm',))}
    df = pd.io.parsers.read_csv(
        filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None,
        sep=',',
    )
    df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
    df.dropna(how="all", inplace=True)  # to drop the empty line at file-end
    X = X = df.loc[:, ['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']].values
    y = df['class label'].values

    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1
    label_dict = {1: 'Setosa', 2: 'Versicolor', 3: 'Virginica'}
    return X, y

def LDA():
    X, y = LoadData()

    # 计算均值向量
    np.set_printoptions(precision=4)
    mean_vectors = []
    for cl in range(1, 4):
        mean_vectors.append(np.mean(X[y == cl], axis=0))
        print('Mean Vector class %s: %s\n' % (cl, mean_vectors[cl - 1]))

    # 计算散度矩阵 S_W 和 S_B
    S_W = np.zeros((4, 4))
    for cl, mv in zip(range(1, 4), mean_vectors):
        class_sc_mat = np.zeros((4, 4))  # scatter matrix for every class
        for row in X[y == cl]:
            row, mv = row.reshape(4, 1), mv.reshape(4, 1)  # make column vectors
            class_sc_mat += (row - mv).dot((row - mv).T)
        S_W += class_sc_mat
    print('within-class Scatter Matrix:\n', S_W)

    overall_mean = np.mean(X, axis=0) #全局均值
    S_B = np.zeros((4, 4))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y == i + 1, :].shape[0]
        mean_vec = mean_vec.reshape(4, 1)  # make column vector
        overall_mean = overall_mean.reshape(4, 1)  # make column vector
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    print('between-class Scatter Matrix:\n', S_B)

    # 计算特征向量和特征值
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # 取前K个特征向量
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True) #排序
    W = np.hstack((eig_pairs[0][1].reshape(4, 1), eig_pairs[1][1].reshape(4, 1))) #取前两个

    # 将样本集X按照W进行映射, X_lda即为新的样本集
    X_lda = X.dot(W)
    return  X_lda

if __name__ == '__main__':
    result = LDA()
    print(result)