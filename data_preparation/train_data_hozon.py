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

#�N���X���Ƃ̔z����P���f�[�^�A�e�X�g�f�[�^�ɕ����āA�t�@�C���ɕۑ�����


#�z����e�X�g�A�P���ɕ�����

test,test_label,train,train_label=vec2train.tensor_2_all()

#(28,28,3)��(3,28,28)�ɕϊ�����֐�convert

def convert(train):
    #��̔z��p��
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
    #����ւ���
    for a in range(28):
        for b in range(28):
            for c in range(3):
                result[c][a][b]=train[a][b][c]

    return result

#�P���f�[�^��(3,28,28)�ɓ���ւ���
train_result=[]
for a in range(1972):
    train_result.append(convert(train[a]))

#�e�X�g�f�[�^��(3,28,28)�ɓ���ւ���
test_result=[]
for a in range(30):
    test_result.append(convert(test[a]))

#�P���f�[�^�A�e�X�g�f�[�^���t�@�C���ɕۑ�����

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