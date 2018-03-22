#label_cov.py

import pickle
import numpy as np

#�t�@�C���ɕۑ����ꂽone-hot���x�����N���X�ԍ����x���Ńt�@�C���ɕۑ�����


#�t�@�C���ɕۑ����ꂽ�������x����ǂݍ���

f=open("train_label","rb")
train_label=pickle.load(f)
f.close()

f=open("test_label","rb")
test_label=pickle.load(f)
f.close()

train_label_cov=[]
test_label_cov=[]


#one-hot���x������A�N���X�ԍ����x���ɕϊ�
for a in range(len(train_label)):
    train_label_cov.append(np.argmax(train_label[a]))
for b in range(len(test_label)):
    test_label_cov.append(np.argmax(test_label[b]))

#���ꂼ��t�@�C���ɕۑ�
f=open("train_label_cov","wb")
pickle.dump(train_label_cov,f)
f.close()

f=open("test_label_cov","wb")
pickle.dump(test_label_cov,f)
f.close()