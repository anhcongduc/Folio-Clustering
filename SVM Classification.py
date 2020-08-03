from __future__ import print_function
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
import operator
import timeit
import math
from PIL import Image
from imutils import paths
import os


def bitmap(img, scale):
    # scale image
    w, h = img.size
    s = scale / (w * h)
    w = int(math.sqrt(s) * w)
    h = int(math.sqrt(s) * h)
    img = img.resize((w, h), Image.ANTIALIAS)

    arr = np.array(img).flatten()
    zero = np.zeros(((scale - w * h) * 3,), dtype=int)
    arr = np.concatenate((arr, zero))

    return arr


def load(scale):
    print("Loading data...")
    # Tạo dữ liệu bitmap và label
    bmp_data = []
    labels = []
    imagePaths = list(paths.list_images('Folio'))

    for (i, imagePath) in enumerate(imagePaths):
        label = imagePath.split(os.path.sep)[-2]
        img = Image.open(imagePath)
        bmp = bitmap(img, scale)
        bmp_data.append(bmp)
        labels.append(int(label))
        print(i)
    flag=1234
    train_bmp_data=[]
    train_labels_data=[]
    test_bmp_data=[]
    test_labels=[]
    count=0
    for i in range(len(labels)):
        if(labels[i]!=flag):
            count+=1
            train_bmp_data.append(bmp_data[i])
            train_labels_data.append(labels[i])
        else:
            test_bmp_data.append(bmp_data[i])
            test_labels.append(labels[i])
        if(count==10):
            count=0
            flag=labels[i]
    print("Load data success!")
    return bmp_data, labels,train_bmp_data,train_labels_data


def process(X, Y, X_test):
    print("SVM...")
    model = SVC(gamma='scale',random_state=32)
    model.fit(X, Y)
    pred = model.predict(X_test)
    return pred
def accuracy(pred_label, Y, labels_num):
    real_label = pred_label
    l = len(pred_label)

    for k in range(0, labels_num):
        vocab  = dict()
        for i in range (0, l):
            if pred_label[i] == k :
                vocab[Y[i]] = vocab.get(Y[i], 0) + 1
        sorted_x = sorted(vocab.items(), key=operator.itemgetter(1))

        for i in range(0, l):
            if pred_label[i] == k:
                real_label[i] = sorted_x[len(sorted_x) - 1][0]

    acc = 0
    for i in range(0,l):
        if (real_label[i]==Y[i]):
            acc = acc + 1
    #print(real_label)
    #print(Y)
    print("==============================")
    print("Accuracy :", (acc/l) *100, "%")


def main():
    start = timeit.default_timer()

    # Chuyển dữ liệu ảnh sang bitmap và scale
    Test_X, Test_Y,Train_X,Train_Y = load(scale = 50*100 )

    print(len(Test_X))
    print(len(Test_Y))
    print(len(Train_X))
    print(len(Train_Y))
    # Chạy Kmeans và gán nhãn
    pred_label = process(Train_X,Train_Y, Test_X)
    # Tính accuracy với nhãn thật
    accuracy(pred_label, Test_Y, labels_num = 32)
    print("Completeness :", completeness_score(Test_Y,pred_label)*100, "%")
    print("Homogeneity :", homogeneity_score(Test_Y,pred_label)*100, "%")

    stop = timeit.default_timer()
    print("Time :", stop - start, "s")


if __name__ == "__main__":
    main()