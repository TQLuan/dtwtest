import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial import distance
from pyts.classification import KNeighborsClassifier

# Import data and augment for labeling
df = pd.read_csv(os.getcwd() + '/lumenMaintenance.csv')
df_with_label = df
df_time = pd.DataFrame(df.iloc[:, 12:])

#Interpolation
dataframe = df_time.interpolate(method='linear', axis=1, inplace=False)
dataframe_with_label = dataframe
# dataframe = df_time.interpolate(method='polynomial', order=3,axis=1,inplace=False)
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(dataframe.values), columns=dataframe.columns, index=dataframe.index)


def test_elbow():
    distortions = []
    for k in range(1, 10):
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(16, 8))
    plt.plot(range(1, 10), distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def k_means():
    m, n = df.shape
    k = 3
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)

    labels = kmeans.labels_
    dataframe_with_label['Pattern'] = labels.T
    dataframe_with_label.to_csv(os.getcwd() + '/labeled_dataset.csv', index=False)
    center_predict = kmeans.cluster_centers_
    fig, ax = plt.subplots(k, 1, sharex=True, sharey=True)
    for i in range(k):
        data_temp = df.loc[labels == i, :]
        m_temp, n_temp = data_temp.shape
        for j in range(m_temp):
            ax[i].plot(list(data_temp.iloc[j, :]), color='r')
            plt.xticks(range(n_temp), data_temp.columns)
            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(15))
        ax[i].plot(range(n_temp), center_predict[i], color='b')
    plt.show()


# test_elbow()
# k_means()
# min_max_scaling()

#Interpolation
# data for dtw-knn
df_knn = pd.read_csv(os.getcwd() + '/labeled_dataset.csv')
#df_knn = df_knn.iloc[0:400]
train_set = df_knn.sample(frac=0.6, random_state=0, axis=0)
test_set = df_knn[~df_knn.index.isin(train_set.index)]
train_data = train_set.iloc[:, 0:154]
train_labels = train_set['Pattern']
test_data = test_set.iloc[:, 0:154]
test_labels = test_set['Pattern']

train_data = pd.DataFrame(scaler.fit_transform(train_data.values), columns=train_data.columns, index=train_data.index)
test_data = pd.DataFrame(scaler.fit_transform(test_data.values), columns=test_data.columns, index=test_data.index)

classifier = KNeighborsClassifier(n_neighbors=1, metric='dtw')
classifier.fit(train_data, train_labels)
labels_pred = classifier.predict(test_data)
print(confusion_matrix(test_labels, labels_pred))
print(classification_report(test_labels, labels_pred))


def predict(K, train_data, train_labels, test_data, test_labels, labels_name):
    i = 0
    accuracy = 0
    predict_labels = []
    for test in test_data.iterrows():
        t_dis = []
        for train in train_data.iterrows():
            dis = 0  # dtw(test.to_numpy(), train.to_numpy())  # dtw计算距离
            t_dis.append(dis)  # 距离数组
        # KNN算法预测标签
        nearest_series_labels = np.array(train_labels[np.argpartition(t_dis, K)[:K]]).astype(int)
        preditc_labels_single = np.argmax(np.bincount(nearest_series_labels))
        predict_labels.append(preditc_labels_single)
        # 计算正确率
        if preditc_labels_single == test_labels[i]:
            accuracy += 1
        i += 1
    print('The accuracy is %f (%d of %d)' % ((accuracy / test_data.shape[0]), accuracy, test_data.shape[0]))
    plt.plot(test_labels, predict_labels, labels_name)  # 绘制混淆矩阵
    return accuracy / test_data.shape[0]


# labels_name = [0,1,2]


# predict(1, train_data, train_labels, test_data, test_labels, labels_name)

# def knn_with_dtw():

# X = df_knn.iloc
cm = confusion_matrix(test_labels, labels_pred)  # 由原标签和预测标签生成混淆矩阵
#plt.imshow(cm, interpolation='nearest')
plt.matshow(cm, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
cb = plt.colorbar()  # 颜色标签
cb.ax.tick_params(labelsize=14)  # 设置色标刻度字体大小。
for x in range(3):
    for y in range(3):
        plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center', fontsize=14)
num_x = np.array(range(3))
num_y = np.array(range(3))
labels_name = [0,1,2]
plt.xticks(num_x, labels_name, fontsize=16)  # 将标签印在x轴坐标上
plt.yticks(num_y, labels_name, fontsize=16)
plt.ylabel('True Area', fontsize=22)  # 坐标轴标签
plt.xlabel('Predicted Area', fontsize=22)  # 坐标轴标签
plt.title('Confusion Matrix', fontsize=22)
plt.show()