import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2


dir_path = './imgs/'


class ImageGetter(object):
    img_ind = 0

    def __init__(self):
        self.img_sub = rospy.Subscriber(
            'camera3/image_color/compressed', CompressedImage, self.call_back, queue_size=1)

    def call_back(self, data):
        print data.header.stamp.to_nsec()
        np_arr = np.fromstring(data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(dir_path + str(data.header.stamp.to_nsec()) + '.jpg', image_np)
        cv2.imshow('sadas', image_np)
        cv2.waitKey(2)
        self.img_ind += 1


if __name__ == '__main__':
    rospy.init_node('get_image')
    ig = ImageGetter()
    rospy.spin()
