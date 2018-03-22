#train.py

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

#ファイルに保存された訓練データとテストデータを読み込む

f=open("train_cov","rb")
train_cov=pickle.load(f)
f.close()

f=open("train_label_cov","rb")
train_label=pickle.load(f)
f.close()

f=open("test_cov","rb")
test_cov=pickle.load(f)
f.close()

f=open("test_label_cov","rb")
test_label=pickle.load(f)
f.close()

#精度を確認する関数test

def test():

    ok=0

    for a in range(len(test_cov)):

        x=Variable(np.array([test_cov[a]],dtype=np.float32))

        t=test_label[a]

        out=model.fwd(x)
        ans=np.argmax(out.data)

        if ans==t:
            ok+=1

    result=ok/len(test_cov)

    print(result)

class MyModel(Chain):
    #パラメータ
    def __init__(self):
        super(MyModel,self).__init__(
            cn1=L.Convolution2D(3,20,5),
            cn2=L.Convolution2D(20,50,5),
            l1=L.Linear(800,500),
            l2=L.Linear(500,2),
        )
    #損失関数
    def __call__(self,x,t):
        return F.softmax_cross_entropy(self.fwd(x),t)
    
    #ネットワークの接続、活性化関数
    def fwd(self,x):
        h1=F.max_pooling_2d(F.relu(self.cn1(x)),2)
        h2=F.max_pooling_2d(F.relu(self.cn2(h1)),2)
        h3=F.dropout(F.relu(self.l1(h2)))
        return self.l2(h3)


#学習の準備、SGDではなく、Adamを使う
model=MyModel()
optimizer=optimizers.Adam()
optimizer.setup(model)

#インプットする訓練データをChainer用に準備
x=Variable(np.array(train_cov,dtype=np.float32))
t=Variable(np.array(train_label,dtype=np.int32))

#バッチ学習、並行して精度も確認
for a in range(10):

    model.cleargrads()
    loss=model(x,t)
    loss.backward()
    optimizer.update()
    if a%1==0:
        test()