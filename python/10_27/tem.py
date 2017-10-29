#!/usr/bin/env python
# -*- coding:utf-8 -*-

import glob
import cv2
imgs = glob.glob('./*/*.jpg')
count = 0
for img in imgs:
    a = cv2.imread(img)
    cv2.imwrite('10_27_'+str(count)+'.jpg', a)
    count +=1
