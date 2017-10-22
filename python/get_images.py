import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2


dir_path = '/home/dhc/imgs'


class ImageGetter(object):
    img_ind = 0

    def __init__(self):
        self.img_sub = rospy.Subscriber(
            'camera1/image_color/compressed', CompressedImage, self.call_back, queue_size=1)

    def call_back(self, data):
        np_arr = np.fromstring(data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(dir_path + str(self.img_ind) + '.jpg', image_np)
        self.img_ind += 1


if __name__ == '__main__':
    rospy.init_node('get_image')
    ig = ImageGetter()
    rospy.spin()
