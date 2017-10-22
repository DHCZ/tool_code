#!/usr/bin/env python
# -*- coding:utf-8 -*-

import glob
data_path='./shanghai'
imgs=glob.glob(data_path+'*.jpg')
imgs.sort()
with open(data_path+'img.lst', 'w') as f:
    for x in imgs:
        f.write("file '{}'\n".format(x))
