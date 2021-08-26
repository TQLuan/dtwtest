import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import h5py
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial import distance
from pyts.classification import KNeighborsClassifier

# Import data and augment for labeling
#df = pd.read_csv(os.getcwd() +'/LumenMaintenance.csv')
#df_with_label = df
#df_time = pd.DataFrame(df.iloc[:, 12:])


#Interpolation
#dataframe = df_time.interpolate(method='linear', axis=1, inplace=False)
#dataframe_with_label = dataframe
## dataframe = df_time.interpolate(method='polynomial', order=3,axis=1,inplace=False)
scaler = MinMaxScaler()
#df = pd.DataFrame(scaler.fit_transform(dataframe.values), columns=dataframe.columns, index=dataframe.index)



#def test_elbow():
#    distortions = []
#    for k in range(1, 7):
#        kmeanModel = TimeSeriesKMeans(n_clusters=k, metric='euclidean')
#       kmeanModel.fit(df)
#        distortions.append(kmeanModel.inertia_)

#    plt.figure(figsize=(16, 8))
#    plt.plot(range(1, 7), distortions, 'bx-')
#    plt.xlabel('k')
#    plt.ylabel('Distortion')
#    plt.show()


#def k_means():
#    m, n = df.shape
#    k = 4
#    kmeans = TimeSeriesKMeans(n_clusters=k, metric='euclidean')
#    kmeans.fit(df)

#    labels = kmeans.labels_
    #print(labels)
    #dataframe_with_label['Pattern'] = labels.T
    #dataframe_with_label.to_csv(os.getcwd() + '/testk=4_labeled_dataset.csv', index=False)
    #df_with_label['Pattern'] = labels.T
    #df_with_label.to_csv(os.getcwd() + '/uninter_testk=4_labeled_dataset.csv', index=False)
#    center_predict = kmeans.cluster_centers_
#    fig, ax = plt.subplots(k, 1, sharex=True, sharey=True)
#    for i in range(k):
#        data_temp = df.loc[labels == i, :]
#        m_temp, n_temp = data_temp.shape
#        for j in range(m_temp):
#            ax[i].plot(list(data_temp.iloc[j, :]), color='r')
#            plt.xticks(range(n_temp), data_temp.columns)
#            plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(15))
#        ax[i].plot(range(n_temp), center_predict[i], color='b')
#    #fig.ylabel('Scaled Degradation Percentage')  # 坐标轴标签
#    fig.text(0.06, 0.5, 'Scaled Degradation Percentage', ha='center', va='center', rotation='vertical')
#    plt.xlabel('Time in Hour')  # 坐标轴标签
#    plt.show()
#    plt.savefig('clusterk=3.png')

def knndtw():
    # Interpolation
    # data for dtw-knn
    df_knn = pd.read_csv(os.getcwd() + '/knn_dataset.csv')
    # df_knn = df_knn.iloc[0:400]
    train_set = df_knn.sample(frac=0.6, random_state=0, axis=0)
    test_set = df_knn[~df_knn.index.isin(train_set.index)]
    train_data = train_set.iloc[:, 0:156]
    train_labels = train_set['Pattern']
    test_data = test_set.iloc[:, 0:156]
    test_labels = test_set['Pattern']

    train_data = pd.DataFrame(scaler.fit_transform(train_data.values), columns=train_data.columns,
                              index=train_data.index)
    test_data = pd.DataFrame(scaler.fit_transform(test_data.values), columns=test_data.columns, index=test_data.index)

    #classifier = KNeighborsClassifier(n_neighbors=1, metric='dtw')
    classifier = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric='dtw')
    classifier.fit(train_data, train_labels)
    labels_pred = classifier.predict(test_data)
    print(confusion_matrix(test_labels, labels_pred))
    print(classification_report(test_labels, labels_pred))

    # labels_name = [0,1,2]

    # predict(1, train_data, train_labels, test_data, test_labels, labels_name)

    # def knn_with_dtw():

    # X = df_knn.iloc
    cm = confusion_matrix(test_labels, labels_pred)  # 由原标签和预测标签生成混淆矩阵
    # plt.imshow(cm, interpolation='nearest')
    plt.matshow(cm, cmap=plt.cm.Blues)  # 画混淆矩阵，配色风格使用cm.Blues
    cb = plt.colorbar()  # 颜色标签
    cb.ax.tick_params(labelsize=14)  # 设置色标刻度字体大小。
    for x in range(4):
        for y in range(4):
            plt.annotate(cm[x, y], xy=(y, x), horizontalalignment='center', verticalalignment='center', fontsize=14)
    num_x = np.array(range(4))
    num_y = np.array(range(4))
    labels_name = [0, 1, 2, 3]
    plt.xticks(num_x, labels_name, fontsize=16)  # 将标签印在x轴坐标上
    plt.yticks(num_y, labels_name, fontsize=16)
    plt.ylabel('True Area', fontsize=22)  # 坐标轴标签
    plt.xlabel('Predicted Area', fontsize=22)  # 坐标轴标签
    plt.title('Confusion Matrix', fontsize=22)
    plt.show()

#def interpolation():
    #df = pd.read_csv(os.getcwd() + '/univariate_cropped.csv')
    #df_time = pd.DataFrame(df.iloc[:, 12:])
    #dataframe = df_time.interpolate(method='linear', axis=1, inplace=False)
    #dataframe.to_csv(os.getcwd() + '/cropped_inter.csv', index=False)

    #df1 = pd.read_csv(os.getcwd() + '/univariate_translated_1kh.csv')
    #df_time1 = pd.DataFrame(df1.iloc[:, 12:])
    #dataframe1 = df_time1.interpolate(method='linear', axis=1, inplace=False)
    #dataframe1.to_csv(os.getcwd() + '/1kh_inter.csv', index=False)

    #df2 = pd.read_csv(os.getcwd() + '/univariate_translated_2kh.csv')
    #df_time2 = pd.DataFrame(df2.iloc[:, 12:])
    #dataframe2 = df_time2.interpolate(method='linear', axis=1, inplace=False)
    #dataframe2.to_csv(os.getcwd() + '/2kh_inter.csv', index=False)
#test_elbow()
#k_means()
# min_max_scaling()
knndtw()
#interpolation()