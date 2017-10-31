from itertools import product
import sys
import numpy as np
import cv2
from os.path import join
import ConfigParser
from mot_tracking import pipe

from obj_detect.tester import ObjectDetectionTester

class BoundFitter(object):
    # theta, b
    ps_range = [0.2, 20]

    def __init__(self):
        self.bound_param = None
        self.pts = None
    def fit(self, box, score_img, seg_img, vis=False, freq=20):
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        if self.bound_param is None:
            # TODO
            # find better initial search space
            theta = np.linspace(-1, 1, freq)
            if (box[0] + box[2]) / 2. < 640:
                theta = np.linspace(-np.pi/4, np.pi/4, freq)
                b = np.linspace(box_w, 2 * box_w, freq)
            else:
                theta = np.linspace(-np.pi/4, np.pi/4, freq)
                b = np.linspace(-box_w, box_w, freq)
        else:
            theta = np.linspace(self.bound_param[0] - self.ps_range[0], self.bound_param[0] + self.ps_range[0], freq)
            b = np.linspace(self.bound_param[1] - self.ps_range[1], self.bound_param[1] + self.ps_range[1], freq)
        self.bound_param, pts, score = self.search(box, score_img, seg_img, [theta, b], vis=vis)
        return self.bound_param, pts, score

    @staticmethod
    def search(box, score_img, seg_img, param_grid, vis=False):
        if vis:
            cv2.rectangle(seg_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        max_p, max_score, max_score_pts = None, -1, None
        for p in product(param_grid[0], param_grid[1]):
            score, pts = BoundFitter.score_of_line(box, score_img, p)
            if vis:
                img = np.array(seg_img)
                cv2.line(img,
                            (int(pts[0][0]), int(pts[1][0])),
                            (int(pts[0][-1]), int(pts[1][-1])),
                            color=(0, 0, 255), thickness=2)
                cv2.imshow('', img)
                cv2.waitKey(10)
            if score > max_score:
                max_score = score
                max_score_pts = pts
                max_p = p
        # TODO
        # remov bad initial state
        if max_score_pts is None or len(max_score_pts[0]) < 5:
            return None, max_score_pts, max_score
        return max_p, max_score_pts, max_score



    @staticmethod
    def score_of_line(box, seg, p):
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        k = np.tan(p[0])
        hx = np.linspace(0, box_h, 15)
        wy = k * hx + p[1]
        # choose the point in bbox
        index = wy<  box_w
        index2 = wy[index] > 0
        hx = hx[index][index2]
        wy = wy[index][index2]
        if len(hx) <= 3 or abs(p[0])>np.pi/4:
            return  -1, []
        pts = [wy + box[0], hx + box[1]]
        k2 = 0 if k==0 else -1/k
        score = 0
        left_pts, right_pts = [[np.array([]), np.array([])], [np.array([]),np.array([])]]
        for grid in (1, 2, ):
            left_pts[0] = np.clip(wy + box[0]-grid, 0, 1279).astype(int)
            left_pts[1] = np.clip(hx+ box[1]-grid*k2 , 0, 719)
            right_pts[0] = np.clip(wy + box[0] + grid , 0, 1279)
            right_pts[1] = np.clip(hx + box[1] + grid * k2 , 0, 719)
            left_sc, right_sc = seg[left_pts[1].astype(int), left_pts[0].astype(
                int)], seg[right_pts[1].astype(int), right_pts[0].astype(int)]
            sum_result = left_sc + right_sc
            zeros_num = len(sum_result[sum_result==0]) / float(len(sum_result))
            score += abs(k *zeros_num )   
        # score function
        # k*num_of_zeros
        # mix k to prevent vertical
        #sum_result = left_sc + right_sc
        #zeros_num = len(sum_result[sum_result==0])
        #score = abs(k * zeros_num )
        return score, pts

def color2label(img):
    img = np.array(img, np.uint8) 
    score_img = np.ones((img.shape[0], img.shape[1]))*10
    color =((128,64, 129), (70, 70, 70))
    span = 10                                                                                                                                    
    for i in range(2):                                                                                                                                                                          
         c = np.array(list(color[i]))
         mask = cv2.inRange(img, tuple((c-span).tolist()), tuple((c+span).tolist()))                                                                                                                   
         mask = mask == 255
         if i==1:
            img[mask] =(142, 0, 0)   
         score_img[mask] = -1 if i==0 else 1
    return score_img, img


def test_main():
    bfs = {}
    forward_config_path = '/home/dhc/workspace/octopus/ros/src/perception/perception_master/library/octopus-ia-objdetect/config/shanqi_forward_cams.config'
    forward_cfg = ConfigParser.ConfigParser()
    forward_cfg.read(forward_config_path)
    track_config_path = '/home/dhc/workspace/octopus/ros/src/perception/perception_master/library/octopus-ia-tracking/config/tracking.config'
    obj = ObjectDetectionTester(cfg=forward_cfg)
    trk = pipe.Pipeline(track_config_path)
    dir_path = '/home/dhc/dhc'
    for index in range(2000, 4000):
        img_path = join(dir_path, 'imgs'+str(index) + '.jpg')
        img = cv2.imread(img_path)
        if img is None:
            continue
        ori_img = img[:720, :1280]
        seg_img = img[720:, :1280]
        score_img, seg_img = color2label(seg_img)
        dets = obj.test_batch([ori_img, ori_img, ori_img], visualize=False)['bboxes_percent'][0]
        res = []
        for d in dets:
            res.extend(d)
        trks = trk.process(ori_img, res, [])
        dets = [[b[0] * 1280, b[1] * 720, b[2] * 1280, b[3] * 720, trks[i]] for i, b in enumerate(res)]
        for b in dets:
            if b[-1] not in bfs:
                bfs[b[-1]] = BoundFitter()
            bf = bfs[b[-1]]
            p, pts, score = bf.fit(b, score_img, seg_img, vis=False)
            cv2.namedWindow('result')
            if p is not None:
                cv2.line(seg_img,
                            (int(pts[0][0]), int(pts[1][0])),
                            (int(pts[0][-1]), int(pts[1][-1])),
                            color=(0, 0, 255), thickness=2)
            cv2.rectangle(seg_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            cv2.imshow('result', seg_img)
            cv2.waitKey(1)
            # import ipdb
            # ipdb.set_trace()
            # a = 1


if __name__ == '__main__':
    test_main()
