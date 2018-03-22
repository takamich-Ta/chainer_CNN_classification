#label_cov.py

import pickle
import numpy as np

#ファイルに保存されたone-hotラベルをクラス番号ラベルでファイルに保存する


#ファイルに保存された正解ラベルを読み込む

f=open("train_label","rb")
train_label=pickle.load(f)
f.close()

f=open("test_label","rb")
test_label=pickle.load(f)
f.close()

train_label_cov=[]
test_label_cov=[]


#one-hotラベルから、クラス番号ラベルに変換
for a in range(len(train_label)):
    train_label_cov.append(np.argmax(train_label[a]))
for b in range(len(test_label)):
    test_label_cov.append(np.argmax(test_label[b]))

#それぞれファイルに保存
f=open("train_label_cov","wb")
pickle.dump(train_label_cov,f)
f.close()

f=open("test_label_cov","wb")
pickle.dump(test_label_cov,f)
f.close()