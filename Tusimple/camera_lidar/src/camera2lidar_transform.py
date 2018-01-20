import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CompressedImage, PointCloud2
from utils import load_params, project_pts
import cv2
import pandas as pd
import numpy as np
from laser_geometry import LaserProjection
import rospy
from obj_detect.tester import ObjectDetectionTester
import ConfigParser
from mot_tracking import pipe

class ImageGetter(object):
    def __init__(self, cam_id=1):
        self.img_time = 0
        self.pd_time = 0
        self.img = None
        self.lidar_pts = None

        self.img_list = []
        self.lidar_pts_list = []

        self.img_time_list = []

        self.img_sub = rospy.Subscriber(
            'camera{}/image_color/compressed'.format(cam_id), CompressedImage, self.get_img, queue_size=1)

        lidar_sub = '/rslidar_points'
        lidar_sub = '/points_segmented'
        self.lidar_sub = rospy.Subscriber(
           lidar_sub, PointCloud2, self.get_pd, queue_size=1)

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
        gen = pc2.read_points(point_cloud, skip_nans=True)
        self.lidar_pts_list.append(gen)
        # if len(self.lidar_pts_list) >= 3:
        #     del self.lidar_pts_list[0]
        #     del self.pd_time_list[0]
        self.lidar_pts = np.array([list(p) for p in gen])

    def get_img_lidar(self):
        if len(self.img_time_list) == 0 or len(self.img_list) == 0 or len(self.img_time_list) != len(self.img_list):
            return None, None, 0
        time_list = abs(self.pd_time - np.array(self.img_time_list))
        i = np.argmin(time_list)
        return self.img_list[i], self.lidar_pts, time_list[i]


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
    def __init__(self, params_yaml, depth_range=(0, 200), height_range=(-20, -2)):


        self.depth_range = depth_range
        self.height_range = height_range
        self.color_map = ColorMap((255, 0, 0), (0, 0, 255), 0, 200)

        self.cam_mats = {"cam1": None, "cam2": None}
        self.dist_coefs = {"cam1": None, "cam2t": None}
        self.rect_rot = {"cam1": None, "cam2": None}
        self.color_map = ColorMap((255, 0, 0), (0, 0, 255), 0, 100)
        self.proj_mats = {"cam1": None, "cam2": None}
        self.undistortion_map = {"cam1": None, "cam2": None}
        self.rectification_map = {"cam1": None, "cam2": None}

        self.cam_mats['cam1'], self.dist_coefs['cam1'], self.lidar_rot = load_params(
            params_yaml)

    def rectify(self, img, lidar_pts):
        if lidar_pts is None:
            return None
        self.height, self.width = img.shape[: 2]
        cam_pts_3d, cam_pts_2d, lidat_pts_proj = project_pts(
            lidar_pts, self.lidar_rot, self.cam_mats['cam1'], self.dist_coefs['cam1'], self.depth_range, self.height_range, self.width, self.height)
        for i, pt in enumerate(cam_pts_2d):
            cv2.circle(img, (int(pt[0]), int(
                pt[1])), 2, self.color_map[lidar_pts[i,4]], thickness=3, lineType=8, shift=0)
        return img, cam_pts_2d, cam_pts_3d, lidat_pts_proj

    def get_location(self, bbox_list, cam_pts_2d, cam_pts_3d, lidar_pts):
        '''
        get the location of bbox by point cloud
        TODO: we choos the point in bbox to get location, However, when a large trunk  in the edge of image, 
              the depth will not change.
              may be we should use camera 6 ,7
        '''
        bbox_depth_list=[]
        cam_pts_2d = np.array(cam_pts_2d)
       
        for bbox in bbox_list:
            ind1 = cam_pts_2d[:,0]>bbox[0]
            ind2 = cam_pts_2d[:,1]>bbox[1]
            ind3 = cam_pts_2d[:,0]<bbox[2]
            ind4 = cam_pts_2d[:,1]<bbox[3]
            bbox_depth = cam_pts_3d[ind1&ind2&ind3&ind4][:, 2]
            lidar_pts_valid = lidar_pts[ind1&ind2&ind3&ind4][:]
            seg_class = set(lidar_pts_valid)
            if len(bbox_depth) == 0 or len(seg_class) == 0:
                # TODO handle the conner case
                bbox_depth_list.append(np.median(bbox_depth)) 
            else:
                depth_list = [bbox_depth[lidar_pts_valid== seg] for seg in seg_class]
                depth_list_median = [np.median(depth)  for depth in depth_list]
                bbox_depth_list.append(min(depth_list_median))
        return bbox_depth_list


def init_detector():
    forward_config_path = '/home/hengchen.dai/workspace/octopus2/src/perception/master/library/octopus-ia-objdetect/config/shanqi_forward_cams.config'
    forward_cfg = ConfigParser.ConfigParser()
    forward_cfg.read(forward_config_path)
    track_config_path = '//home/hengchen.dai/workspace/octopus2/src/perception/master/library/octopus-ia-tracking/config/tracking.config'
    obj = ObjectDetectionTester(cfg=forward_cfg)
    trk = pipe.Pipeline(track_config_path)
    return obj, trk


if __name__ == "__main__":
    # img_file = "/home/hengchen.dai/workspace/tool_code/python/imgs/1516036969670058012.jpg"
    # lidar_file = "/home/hengchen.dai/workspace/tool_code/python/1516036969678430.pcd"

    # img = cv2.imread(img_file)
    # lidar_pts = pd.read_csv(lidar_file, header = None, sep = ' ', skiprows=11).values[:, 0: 3]
    rospy.init_node('canera_lidat_transform')
    ig = ImageGetter(cam_id=1)
    camera_trans = camera2lidar_trainsform('./config/cam1.yaml')

    obj, trk= init_detector()
    while True:
        # get image
        img, lidar, diff_time = ig.get_img_lidar()

        #print diff_time / 1e6
        if img is None:
            continue
        # get_bbox
        dets = obj.test_batch([img, img, img], visualize=False)['bboxes_percent'][0]
        res = []
        for d in dets:
            res.extend(d)
        trks = trk.process(img, res, [])
        dets = [[b[0] * 1280, b[1] * 720, b[2] * 1280, b[3] * 720, trks[i]] for i, b in enumerate(res)]
        
        #project ldiar pts to image
        img, cam_pts_2d, cam_pts_3d, lidar_pts = camera_trans.rectify(img, lidar)
        bbox_depth_list = camera_trans.get_location(dets,cam_pts_2d, cam_pts_3d, lidar_pts)
        for b,depth in zip(dets, bbox_depth_list):
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.putText(img, str(depth), (int(b[0]), int(b[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),2,cv2.LINE_AA )
        cv2.imshow('camera_lidar', img)
        key = cv2.waitKey(1)
        if key == 27:  # press esc
            break
    rospy.spin()
