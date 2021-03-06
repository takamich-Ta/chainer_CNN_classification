#vec2train.py

import numpy as np
from PIL import Image
import random
import gazo2vec

#訓練データ、訓練教師データ、テストデータ、テスト教師データを用意する関数

def tensor_2_all():
    #901,1101,30
    test=30
    #テスト30枚分、latteが15
    latte_test=int(test/2)
    #otherが15
    other_test=int(test/2)
    latte_train=901-latte_test
    other_train=1101-other_test

    latte_r, other_r=gazo2vec.jpg_2_tensor()
    #テスト用データ、テスト用データラベル、学習用データ、学習用データラベル
    test_r=[]
    test_rabels=[]
    train_r=[]
    train_rabels=[]

    #それぞれに保存、データ、ラベル
    for a in range(latte_test):
        test_r.append(latte_r[a])
        test_rabels.append([0,1])
    for a in range(other_test):
        test_r.append(other_r[a])
        test_rabels.append([1,0])
    for a in range(latte_train):
        train_r.append(latte_r[a+latte_test])
        train_rabels.append([0,1])
    for a in range(other_train):
        train_r.append(other_r[a+other_test])
        train_rabels.append([1,0])

    test_r=np.array(test_r)
    test_rabels=np.array(test_rabels)
    train_r=np.array(train_r)
    train_rabels=np.array(train_rabels)

    #データ、ラベルを返す、テスト用、学習用
    return test_r,test_rabels,train_r,train_rabels