from __future__ import print_function
from sklearn.cluster import Birch
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
import numpy as np
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

    print("Load data success!")
    return bmp_data, labels


def process(X, labels_num):
    print("Clustering using Birch")
    brc = Birch(branching_factor=20, n_clusters=32, threshold=10,compute_labels = True).fit(X)
    pred_label = brc.predict(X)
    return pred_label
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
        if (real_label[i]!=Y[i]):
            acc = acc + 1
    print("==============================")
    print("Accuracy :", (acc/l) *100, "%")


def main():
    start = timeit.default_timer()

    # Chuyển dữ liệu ảnh sang bitmap và scale
    X, Y = load(scale = 50*100 )

    # Chạy Birch và gán nhãn
    pred_label = process(X, labels_num = 32)
    # Tính accuracy với nhãn thật
    accuracy(pred_label, Y, labels_num = 32)
    print("Completeness :", completeness_score(Y, pred_label) * 100, "%")
    print("Homogeneity :", homogeneity_score(Y, pred_label) * 100, "%")

    stop = timeit.default_timer()
    print("Time :", stop - start, "s")


if __name__ == "__main__":
    main()