import numpy as np
import cv2
from dataset_store import Dataset

if __name__ == '__main__':
    bag_name, ts_begin, ts_end = ['/mnt/truenas/datasets/v2/2017-01-12-11-23-23', '00:00', '60:00']  # big traffic
    bag = Dataset(bag_name)
    args_dataset = [
        '/camera1/image_color/compressed',
        '/camera3/image_color/compressed',
        '/camera4/image_color/compressed',
    ]
    count = 0
    for sensors in bag.fetch_aligned(
        *args_dataset,
         ts_begin=ts_begin,
         ts_end=ts_end
     ):
        count += 1
        raw_cams = [s[1] for s in sensors[0:5]]
        det_cams = [cv2.imdecode(np.fromstring(raw_cams[i].data, np.uint8),
                                         cv2.IMREAD_COLOR)
                              for i in range(len(raw_cams))]
        if count % 399 == 0:
            cv2.imwrite('./image/cam1/2017-12-28-cam1-{0:07d}.png'.format(count), det_cams[0])
            print count
        if (count-248) % 399 == 0:
            cv2.imwrite('./image/cam3/2017-12-28-cam3-{0:07d}.png'.format(count), det_cams[1])
        cv2.waitKey(2)
