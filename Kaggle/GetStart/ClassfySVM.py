# -*- coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
# %matplotlib inline

# 加载数据
labeled_images = pd.read_csv('D:\Python\Practice\scikit-learning\Python_skiLearn\Kaggle\GetStart\\train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)


#显示一个图片样本
i=1
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
print len(img[0])
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
plt.show()

# plt.hist(train_images.iloc[i])



# 将所有像素变为黑或白
# 将像素点值大于0的都置为1
test_images[test_images>0]=1
train_images[train_images>0]=1

print test_images.shape
# 运用svm进行分类训练
clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
score_result = clf.score(test_images,test_labels)
print score_result


#将训练得到的模型运用到分类预测上
test_data=pd.read_csv('D:\Python\Practice\scikit-learning\Python_skiLearn\Kaggle\GetStart\\test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:5000])

# 通过panda将结果数据写到csv文件
df = pd.DataFrame(results)
df.index.name = 'ImageId'
df.index += 1
df.columns = ['Label']
df.to_csv('D:\Python\Practice\scikit-learning\Python_skiLearn\Kaggle\GetStart\\results.csv', header=True)

