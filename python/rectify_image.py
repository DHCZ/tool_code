import cv2
import numpy as np

class ImageRectify(object):

    def __init__(self, camera_param=None):
        M1 = np.array([738.3165, 0., 657.5385, 0., 736.3515, 357.2885, 0., 0., 1.0])
        M1=M1.reshape(3,3)
        D1 =np.array([-0.1036, 0.0790, -0.0023, 0.0034])
        D1 =np.array([0.0, 0.0, 0.00, 0.00, -0.00])
        R = np.array([ 0.03204101, -0.29673793,  0.95442129,-0.99924627,-0.03044799,  0.0,0.0, 0.0,1.0,])
        R=R.reshape(3,3)
        T = np.array( [-2.82109741, 0.68336604,  1.524783])
        self.camera_param = {}
        self.camera_param['M1'] = M1
        self.camera_param['D1'] = D1
        self.camera_param['M2'] = M1
        self.camera_param['D2'] = D1
        self.camera_param['R']  = R
        self.camera_param['T']  = T
        self.pts = None
    
    def rectify_image(self, image):
        img_size=(image.shape[1],image.shape[0])
        M1, D1, M2, D2, R, T = self.camera_param['M1'], self.camera_param['D2'], self.camera_param['M2'], self.camera_param['D2'], self.camera_param['R'], self.camera_param['T']
        R1, R2, P1, P2, Q,_,_=cv2.stereoRectify(cameraMatrix1=M1, distCoeffs1=D1,cameraMatrix2= M2,distCoeffs2=D2,imageSize= img_size, R=R, T=T, flags=cv2.CALIB_ZERO_DISPARITY,alpha=-1,newImageSize=img_size)
        print R
        print R1
        map11,map12=cv2.initUndistortRectifyMap(cameraMatrix=M1, distCoeffs=D1, R=R, newCameraMatrix=P1, size=img_size, m1type=cv2.CV_16SC2)

        img1r=cv2.remap(src=image, map1=map11, map2=map12, interpolation=cv2.INTER_LINEAR)
        cv2.imshow('rectify', img1r)
        cv2.waitKey()


if __name__ == '__main__':
    imagerec = ImageRectify()
    img = cv2.imread('./imgs/100.jpg')
    imagerec.rectify_image(img)


