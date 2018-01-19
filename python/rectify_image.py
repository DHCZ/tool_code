mport cv2
import numpy as np

class ImageRectify(object):

    def __init__(self, camera_param=None):
        M1 = np.array([1848.4, 0., 1012.8, 0., 1848.0, 745.64, 0., 0., 1.0])
        M1=M1.reshape(3,3)
        D1 =np.array([-0.1110, 0.244, -0.0012, 0.00092, -0.311])
        R = np.array([ 1.0000, -0.0047,-0.0058, 0.0047, 1.0000, -0.0018, 0.0058, 0.0018, 1.000])
        R=R.reshape(3,3)
        T = np.array( [-499.8443, 5.3435,  0.8954])
        M1 = np.array([1840.9, 0., 978.2887, 0., 1840.7, 742.1273, 0., 0., 1.0])
        M1=M1.reshape(3,3)
        D1 =np.array([-0.1005, 0.1635, -0.0019, -0.0024, -0.0831])
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

        img=cv2.remap(src=image, map1=map11, map2=map12, interpolation=cv2.INTER_LINEAR)
        img = np.concatenate([img1r, img2r], axis=1)
        for h in range(0, height, 50):
            cv2.line(img, (0, h), (2*width-1, h), color=(0, 255, 0), thickness=1)
        cv2.imshow('rectify', img)
        cv2.waitKey()


if __name__ == '__main__':
    imagerec = ImageRectify()
    img = cv2.imread('./imgs/100.jpg')
    imagerec.rectify_image(img)


