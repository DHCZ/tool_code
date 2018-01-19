import yaml
import numpy as np
import cv2

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat

def opencv_matrix_representer(dumper, mat):
    mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    return dumper.represent_mapping(u"tag:yaml.org, 2002:opencv-matrix", mapping)

def load_params(file_name):
    with open(file_name, 'r') as f:
        yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)
        yaml.add_representer(np.ndarray, opencv_matrix_representer)
        yaml_data = yaml.load(f.read())
        intrinsic = np.array(yaml_data['K'])
        distort = np.array(yaml_data['Distortion'])
        c2l_tf = np.array(yaml_data['TransformationFromCamToLiDAR'])
        l2c_tf = np.linalg.inv(c2l_tf)
    return intrinsic, distort, l2c_tf

def project_pts(lidar_pts, l2c_tf, intrinsic, distort, depth_range=(0, 50), height_range=[-0.5, 10],  x_limit=1280, y_limit=720):
    
    lidar_pts = np.concatenate((lidar_pts[:, 0: 3], np.ones((lidar_pts.shape[0], 1))), 1)
    cam_pts_3d = np.dot(l2c_tf, lidar_pts.T)
    ind1 = cam_pts_3d[2, :] > depth_range[0]
    ind2 = cam_pts_3d[2, :] < depth_range[1]
    #ind3 = cam_pts_3d[1, :] > height_range[0]
    #ind4 = cam_pts_3d[1, :] < height_range[1]
    cam_pts_3d = cam_pts_3d[0:3: , ind1 & ind2 ].T
    cam_pts_2d, jacobian = cv2.projectPoints(np.array([cam_pts_3d]), np.identity(3), np.zeros((1, 3)), intrinsic, distort)

    cam_pts_2d = cam_pts_2d[:, 0, :]
    I1 = cam_pts_2d[:, 0] < x_limit
    I2 = cam_pts_2d[:, 0] > 0
    I = I1 & I2
    cam_pts_2d = cam_pts_2d[I, :]
    cam_pts_3d = cam_pts_3d[I, :]

    I1 = cam_pts_2d[:, 1] < y_limit
    I2 = cam_pts_2d[:, 1] > 0
    I = I1 & I2
    cam_pts_2d = cam_pts_2d[I, :]
    cam_pts_3d = cam_pts_3d[I, :]

    return cam_pts_3d, cam_pts_2d
