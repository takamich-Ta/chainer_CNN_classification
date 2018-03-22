#gazo2vec.py

import tensorflow as tf
import numpy as np
from PIL import Image
import random

#�摜�̃t�@�C����z��ɂ���֐�

def jpg_2_tensor():

    right_rabels_latte=[]
    right_rabels_other=[]
    #�摜��
    latte_gazou=901
    other_gazou=1101
    gokei=latte_gazou+other_gazou

    #�߂�l�Ƃ��Ĕz�񂪓���
    latte_r=[]
    other_r=[]

    for a in range(latte_gazou):
        latte_r.append(0)
    for a in range(other_gazou):
        other_r.append(0)


    for a in range(latte_gazou):
        right_rabels_latte.append([0,1])
        b=str(a+1)
        #�摜�ǂݍ���
        latte_r[a]=Image.open("./data/latte/l("+b+").jpg")
        a=int(a)
        # �I���W�i���摜�̕��ƍ������擾
        width, height = latte_r[a].size

        img_pixels = []
        for y in range(height):
            for x in range(width):
                # getpixel((x,y))�ō�����x�Ԗ�,�ォ��y�Ԗڂ̃s�N�Z���̐F���擾���Aimg_pixels�ɒǉ�����
                img_pixels.append(latte_r[a].getpixel((x, y)))
        # numpy��array�ɕϊ�����
        latte_r[a]=np.array(img_pixels)
        #(28,28),RGB�̔z��ɂ���
        latte_r[a]=np.reshape(latte_r[a], (28, 28, 3))

    for b in range(other_gazou):
        right_rabels_other.append([1,0])
        c=str(b+1)
        #�摜��ǂݍ���
        other_r[b]=Image.open("./data/other/o("+c+").jpg")
        b=int(b)
    #�I���W�i���摜�̕��ƍ������擾
        width, height = other_r[b].size
        img_pixels = []
        for y in range(height):
            for x in range(width):
        # getpixel((x,y))�ō�����x�Ԗ�,�ォ��y�Ԗڂ̃s�N�Z���̐F���擾���Aimg_pixels�ɒǉ�����
                img_pixels.append(other_r[b].getpixel((x,y)))
        if(not(isinstance(img_pixels[0],int))):

            if(len(img_pixels[0])==3):

                # numpy��array�ɕϊ�����
                other_r[b] = np.array(img_pixels)
                #(28,28),RGB�̔z��ɂ���
                other_r[b]=np.reshape(other_r[b],(28,28,3))
        else:
            for a in range(784):
                img_pixels[a]=[img_pixels[a],1,1]
            #numpy�ɕϊ�����
            other_r[b] = np.array(img_pixels)
            #(28,28),RGB�̔z��ɂ���
            other_r[b]=np.reshape(other_r[b],(28,28,3))
    #���ꂼ��latte,other�̑S�摜���������z��̃��X�g��Ԃ�
    return latte_r,other_r