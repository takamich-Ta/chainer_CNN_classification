#train_data_hozon.py

import numpy as np
import random
import vec2train
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Link,Chain,ChainList
from chainer import optimizers
from chainer import training,utils,Variable
import pickle

#クラスごとの配列を訓練データ、テストデータに分けて、ファイルに保存する


#配列をテスト、訓練に分ける

test,test_label,train,train_label=vec2train.tensor_2_all()

#(28,28,3)を(3,28,28)に変換する関数convert

def convert(train):
    #空の配列用意
    result=[[],[],[]]
    for a in range(28):
        result[0].append([])
        for b in range(28):
            result[0][a].append([])
        result[1].append([])
        for b in range(28):
            result[1][a].append([])
        result[2].append([])
        for b in range(28):
            result[2][a].append([])
    #入れ替える
    for a in range(28):
        for b in range(28):
            for c in range(3):
                result[c][a][b]=train[a][b][c]

    return result

#訓練データを(3,28,28)に入れ替える
train_result=[]
for a in range(1972):
    train_result.append(convert(train[a]))

#テストデータを(3,28,28)に入れ替える
test_result=[]
for a in range(30):
    test_result.append(convert(test[a]))

#訓練データ、テストデータをファイルに保存する

f=open("train_cov","wb")
pickle.dump(train_result,f)
f.close()

f=open("test_cov","wb")
pickle.dump(test_result,f)
f.close()

f=open("train_label","wb")
pickle.dump(train_label,f)
f.close()

f=open("test_label","wb")
pickle.dump(test_label,f)
f.close()