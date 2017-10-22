#!/usr/bin/python
import rosbag
from cv_bridge import CvBridge
import rospy
import cv2
import json
import random
import os
import sys, getopt

def main(argv):
    inputfile = ''
    outputfile = ''
    cam_no = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:c:")
    except getopt.GetoptError:
        print 'test.py -i <bag> -o <output dir> -c <camera no.>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <bag> -o <output dir> -c <camera no.>'
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg
        elif opt in ("-c"):
            cam_no = arg
    return inputfile, outputfile, cam_no

if __name__ =='__main__':
    (path, output, cam_no) = main(sys.argv[1:])
    bridge = CvBridge()
    print "loading bag"
    bag = rosbag.Bag(path)
    print "loading complete"
    print "-"*10
    rospy.Time(1)
    cluster = 0
    num_now = 0
    count = 0
    for msg in bag.read_messages(topics=['/camera'+ cam_no + '/image_color/compressed']):
    # for msg in bag.read_messages(topics=['/stereo/'+ cam_no + '/compressed']):
    	count += 1
        cluster = msg[1].header.seq/5000
        print "In cluster ",cluster
        filepath = os.path.join(output, str(cluster))
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        cv_image = bridge.compressed_imgmsg_to_cv2(msg[1],'passthrough')
        # cv_image = bridge.imgmsg_to_cv2(msg[1],'bgr8')
        cv2.imwrite(os.path.join(filepath+'{:7}.jpg'.format(count)), cv_image)
        # cv2.imwrite(os.path.join(filepath,str(count)+'.jpg'), cv_image)

