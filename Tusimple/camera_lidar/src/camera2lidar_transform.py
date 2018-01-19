import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CompressedImage, PointCloud2
from utils import load_params, project_pts
import cv2
import pandas as pd
import numpy as np
from laser_geometry import LaserProjection
import rospy


class ImageGetter(object):

    def __init__(self):
        self.img_time = 0
        self.pd_time = 0
        self.img = None
        self.lidar_pts = None

        self.img_list = []
        self.lidar_pts_list = []

        self.img_time_list = []
        
        self.img_sub = rospy.Subscriber(
            'camera1/image_color/compressed', CompressedImage, self.get_img, queue_size=1)

        self.lidar_sub = rospy.Subscriber(
            '/points_segmented', PointCloud2, self.get_pd, queue_size=1)
  
    def get_img(self, data):
        
        self.img_time_list.append(data.header.stamp.to_nsec())
        np_arr = np.fromstring(data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.img_list.append(image_np)
        if len(self.img_list) >= 5:
            del self.img_time_list[0]
            del self.img_list[0]

    def get_pd(self, point_cloud):
        self.pd_time = point_cloud.header.stamp.to_nsec()
        gen = pc2.read_points(point_cloud, skip_nans=True, field_names=("x", "y", "z"))
        # gen = pc2.read_points(point_cloud, skip_nans=True)
        # for p in gen:
        #     print p
        # self.lidar_pts_list.append(gen)
        # if len(self.lidar_pts_list) >= 3:
        #     del self.lidar_pts_list[0]
        #     del self.pd_time_list[0]
        self.lidar_pts = np.array([list(p) for p in gen])

    def get_img_lidar(self):
        if len(self.img_time_list) == 0 or len(self.img_list) == 0 or len(self.img_time_list)!=len(self.img_list):
            return None, None, 0
        time_list = abs(self.pd_time - np.array(self.img_time_list))
        i = np.argmin(time_list)
        return self.img_list[i], self.lidar_pts,time_list[i]


class ColorMap:
    start_color = ()
    end_color = ()
    start_map = 0
    end_map = 0
    color_dis = 0
    value_range = 0
    ratios = []
    def __init__(self, start_color, end_color, start_map, end_map):
        self.start_color = np.array(start_color)
        self.end_color = np.array(end_color)
        self.start_map = float(start_map)
        self.end_map = float(end_map) 
        self.value_range = self.end_map - self.start_map
        self.ratio = (self.end_color - self.start_color) / self.value_range

    def __getitem__(self, value):
        value = max(value, self.start_map)
        value = min(value, self.end_map)
        color = self.start_color + (self.ratio * (value - self.start_map))
        return (int(color[0]), int(color[1]), int(color[2]))


class camera2lidar_trainsform:
    def __init__(self, params_yaml):
        
        self.max_depth = 1000
        self.min_depth = 0

        
        self.color_map = ColorMap((255, 0, 0), (0, 0, 255), 0, 100)

        self.cam_mats = {"left": None, "right": None}
        self.dist_coefs = {"left": None, "right": None}
        self.rect_rot = {"left": None, "right": None}
        self.color_map = ColorMap((255, 0, 0), (0, 0, 255), 0, 100)
        self.proj_mats = {"left": None, "right": None}
        self.undistortion_map = {"left": None, "right": None}
        self.rectification_map = {"left": None, "right": None}



        self.cam_mats['left'], self.dist_coefs['left'], self.lidar_rot = load_params(params_yaml)

    

    def rectify(self, img, lidar_pts):
        if lidar_pts is None:
            return None
        self.height, self.width = img.shape[: 2]
        cam_pts_3d, cam_pts_2d = project_pts(lidar_pts, self.lidar_rot, self.cam_mats['left'], self.dist_coefs['left'], (0, 50), (-20, -2), self.width, self.height)
        for i, pt in enumerate(cam_pts_2d):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 2, self.color_map[cam_pts_3d[i, 2]], thickness=3, lineType=8, shift=0)
        return img
    
    def get_location()
    '''
    get the location of bbox by point cloud
    '''
    



if __name__ == "__main__":
    # img_file = "/home/hengchen.dai/workspace/tool_code/python/imgs/1516036969670058012.jpg"
    # lidar_file = "/home/hengchen.dai/workspace/tool_code/python/1516036969678430.pcd"
    
    # img = cv2.imread(img_file)
    # lidar_pts = pd.read_csv(lidar_file, header = None, sep = ' ', skiprows=11).values[:, 0: 3]
    rospy.init_node('canera_lidat_transform')
    ig = ImageGetter()
    camera_trans = camera2lidar_trainsform('./config/cam1.yaml')
    while True:
        img ,lidar, diff_time = ig.get_img_lidar()
        print diff_time/1e6
        key = cv2.waitKey(1)
        if key == 27:# press esc
            break
        img = camera_trans.rectify(img, lidar)
        if img is None:
            continue
        cv2.imshow('temp', img)
    rospy.spin()

