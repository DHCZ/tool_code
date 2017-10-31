'''
This script is used to validate the alignment of two stereo cameras. 
You can visualize the result by setting visualize=True.
'''

import cv2
import numpy as np
import datetime
import time
import string
import os
from PIL import Image
import matplotlib.pyplot as plt
from dataset_store import Dataset
#save depth map into txt



# set parameters
def sgbm_depth(left_image,right_image,seqeunce_name='dhc',visualize=False):

    img1_out='./sgbm/'+seqeunce_name+'/cam1/'
    img2_out = './sgbm/'+seqeunce_name+'/cam2/'
    if not os.path.exists(img1_out):
        os.makedirs(img1_out)
    if not os.path.exists(img2_out):
        os.makedirs(img2_out)

    # SGM parameters
    minDisparity=0
    numDisparities=256
    SADWindowSize=5
    cn = 3
    P1_sg=8 * cn*SADWindowSize*SADWindowSize
    P2_sg=32 * cn*SADWindowSize*SADWindowSize
    preFilterCap=63
    uniquenessRatio=10
    speckleWindowSize=100
    speckleRange=32
    disp12MaxDiff=1

    #set parameter for stereo vision
    M1 = np.array([7600.84983839602, -2.99902176361271, 1363.98027137587,
                    0.000000, 7606.59010258463, 725.727691214881, 0.000000,
                    0.000000, 1.000000])
    M1=M1.reshape(3,3)
    D1 =np.array([-0.0550340193281919, -0.201838612399988, -0.00487395972599783,	0.00412750068646054])

    M2 = np.array([ 7604.52194213316, -5.65932956048017, 1396.85380783994,
                    0.000000, 7610.60877362907, 705.423206525307, 0.000000,
                    0.000000, 1.000000])
    M2=M2.reshape(3,3)
    D2 = np.array([-0.0320233169370680, -0.0383792527839777, -0.00644739641691392, 0.00447193518679759])
    R = np.array([ 0.999997140595277, 0.00225991554062962,
                   0.000782037735025578, -0.00224644495126731,
                   0.999856056670654,-0.0168172359230492,
                   -0.000819930698723269, 0.0168154310310438,
                   0.999858274453379])
    R=R.reshape(3,3).transpose() 
    T = np.array( [-500.588562682978, 4.46368597454194,	3.59227301902774])

    tic = time.time()

    img1 = left_image
    img2 = right_image
    #save image size

    img_size=img1.shape
    img_size=(img_size[1],img_size[0])

    #R1 is the rectified rotation matrix of camera 1
    #R2 is the rectified rotation matrix of camera 2
    #P1 is the rectified projection matrix of camera1
    #P2 is the rectified projection matrix of camera2
    #Q is the rectified reprojection matrix of camera1 and camera2

    #undistortion and rectified image

    R1, R2, P1, P2, Q,_,_=cv2.stereoRectify(cameraMatrix1=M1, distCoeffs1=D1,cameraMatrix2= M2,distCoeffs2=D2,imageSize= img_size, R=R, T=T, flags=cv2.CALIB_ZERO_DISPARITY,alpha=-1,newImageSize=img_size)
    map11,map12=cv2.initUndistortRectifyMap(cameraMatrix=M1, distCoeffs=D1, R=R1, newCameraMatrix=P1, size=img_size, m1type=cv2.CV_16SC2)
    map21,map22=cv2.initUndistortRectifyMap(cameraMatrix=M2, distCoeffs=D2, R=R2, newCameraMatrix=P2, size=img_size, m1type=cv2.CV_16SC2)

    #rectified image from original image
    #img1r is the rectified image from camera 1
    #img2r is the rectified image from camera 2
    print R1
    print P1

    img1r=cv2.remap(src=img1, map1=map11, map2=map12, interpolation=cv2.INTER_LINEAR)
    img2r=cv2.remap(src=img2, map1=map21, map2=map22, interpolation=cv2.INTER_LINEAR)


    x = np.zeros((1,1,2), dtype=np.float32)
    x[0,0,0]=400
    x[0,0,1]=1000
    x=cv2.undistortPoints(x,M1,D1,R=R1,P=P1)
    print x

    #overwrite img1r to img1
    #overwrite img2r to img2
    img1=img1r
    img2=img2r
    tmp=np.hstack((img1,img2))
    # tmp1=cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

    sgbm=cv2.StereoSGBM_create(minDisparity=minDisparity, numDisparities=numDisparities, blockSize=SADWindowSize, P1=P1_sg, P2=P2_sg, disp12MaxDiff=disp12MaxDiff, preFilterCap=preFilterCap, uniquenessRatio=uniquenessRatio, speckleWindowSize=speckleWindowSize, speckleRange=speckleRange)
    #sgbm = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16, SADWindowSize=15)
    #compute disparity map
    disp=sgbm.compute(img1,img2)
    print disp.shape
    #reproject disparity to 3D
    xyz=cv2.reprojectImageTo3D(disparity=disp,Q=Q,handleMissingValues=True)
    xyz=xyz*16

    print xyz[895][524]
    print 'execute time for computing distance is '+str(time.time() - tic)

    #For visualization SGBM depth image purpose only
    if(visualize==True):
        # visualization of alignment
        plt.figure()
        plt.imshow(tmp)
        #plt.plot([600, 1000], [3500, 500])
        #plt.show()

        #plt.imshow(disp)
        #plt.plot([647, 1055], [3500, 500])
        plt.show()

    # return img1,xyz


if __name__=='__main__':
    img_left = cv2.imread('./image/i01_cam1_1508916929.059156.jpg')
    img_right = cv2.imread('./image/i01_cam2_1508916929.075281.jpg')
    sgbm_depth(left_image=img_left, right_image=img_right ,visualize=True)




