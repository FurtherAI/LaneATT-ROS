#!/usr/bin/env python

# model state dict + anchors

# dependencies
# cv2, numpy, torch, torchvision, scipy

# get past import issues
# run test with bag file
# citation before uploading
# onnx version for cpu
from perception_msgs.msg import lanes, lane, point
from sensor_msgs.msg import Image
import lanedetection
from lanedetection.lane import Lane
from lanedetection.laneatt import LaneATT

import cv2
import numpy as np
import torch
from torchvision import transforms
from cv_bridge import CvBridge, CvBridgeError
import time

from imgaug.augmenters import Resize

import sys

import rospy


class LaneDetector:
    def __init__(self, model_path='/home/further/catkin_ws/src/lanedetection/scripts/model_files/model_0015.pt', device='cuda'):
        self.pub = rospy.Publisher('/lanes', lanes, queue_size=10)
        self.sub = rospy.Subscriber('/camera/image_raw', Image, self.img_callback)
        self.rate = rospy.Rate(10)
        self.image = None

        self.bridge = CvBridge()
        # resize
        self.resize = Resize({'height': 360, 'width': 640})
        self.to_tensor = transforms.ToTensor()
        self.times = np.zeros((4, ), dtype=np.float32)
        self.iters = 0
        self.device = torch.device(device)
        self.str_device = device
        self.model_path = model_path
        self.model = LaneATT(anchors_freq_path='/home/further/catkin_ws/src/lanedetection/scripts/model_files/culane_anchors_freq.pt', topk_anchors=1000)  # top 1000 anchors (frequency of ground truth appearance in data)

    def initialize_model(self):
        # load weights, eval mode
        rospy.loginfo('Loading model weights...')
        state = torch.load(self.model_path)
        self.model.load_state_dict(state['model'], strict=True)
        self.model.to(self.device)
        self.model.eval()
        rospy.loginfo('Model successfully loaded')

    def img_callback(self, data):  # since img_callback seems to be called every time the topic is updated, use as little computation as possible here
        self.image = data

    def detect_lanes(self):  # do actual computation of lanes and publish
        # transform img, forward through model, decode, select two closest to center, format_lanes, publish
        t0 = time.time()
        img = self.transform_img(self.image)
        t1 = time.time()
        self.times[0] += t1 - t0

        img = img.to(self.device)
        with torch.no_grad():
            t2 = time.time()
            output = self.model(img, conf_threshold=.5, nms_thres=50., nms_topk=4, device=self.str_device)
            ta = time.time()
            prediction = self.model.decode(output, as_lanes=True)
            t3 = time.time()
        self.times[1] += ta - t2
        self.times[2] += t3 - ta
        t4 = time.time()
        if len(prediction[0]) != 0:  # prediction is list of len batch_size, containing lists of Lane, or empty lists if predicting no lanes
            lns = prediction[0]
            lns = [ln.points for ln in lns]
            for points in lns:
                points[:, 0] *= img.shape[3]  # scale to img width
                points[:, 1] *= img.shape[2]  # scale to img height
            img_center = img.shape[2] // 2
            points = self.select_closest_lanes(lns, img_center)
        else:
            points = [np.array([[-1, -1]], dtype=np.float32), np.array([[-1, -1]], dtype=np.float32)]
        # format into ros msg
        lanes_ = [lane(list(map(point, tuple(l)))) for l in points]  # for each lane, map the point constructor over points. list[2] of 2d np -> list[2] of list of point
        lanes_output = lanes()
        lanes_output.lanes = lanes_
        lanes_output.header.stamp = rospy.Time.now()
        self.pub.publish(lanes_output)
        t5 = time.time()
        self.times[3] += t5 - t4
        self.iters += 1
        print(self.times / self.iters)

    def select_closest_lanes(self, lns, img_center):  # model produces up to 4 lanes, closest two to the center (camera) should be the surrounding lanes
        if len(lns) == 1:
            idx = 0 if lns[0][-3:, 0].mean() > img_center else 1  # this lane is left or right lane
            lns.insert(idx, np.array([[-1, -1]], dtype=np.float32))  # insert value for no lane
            return lns
        lns.sort(key=lambda ln : abs(ln[-3:, 0].mean() - img_center))  # average last 3 x values - img_center (as long as lane is more than 3)
        # two closest lanes, may be different length points, so keep as list
        return lns[:2]  # should be in order left, right because of sorting

    def transform_img(self, img):
        # cv, bgr, resize, normalize (normal process in lane_dataset)
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            # cv2.imwrite('/home/further/camera_raw.jpg', cv_img)  # if you want to go look at what image it's working on
            img = self.resize(image=cv_img)
            img = self.to_tensor(img)
            img = img.unsqueeze(0)  # batch size 1
            return img

        except CvBridgeError as e:
            print(e)


def parse(args):
    if '--device' in args:
        idx = args.index('--device')
        assert args[idx + 1] in ['cpu', 'cuda']
        return args[idx + 1]
    else:
        return 'cuda'

if __name__ == '__main__':
    args = rospy.myargv(argv=sys.argv)
    device = parse(args)
    rospy.init_node('lane_detection', anonymous=True)
    lane_det = LaneDetector(model_path='/home/further/catkin_ws/src/lanedetection/scripts/model_files/model_0015.pt', device=device)
    lane_det.initialize_model()
    try:
        while not rospy.is_shutdown():
            if lane_det.image is not None:
                lane_det.detect_lanes()
                lane_det.rate.sleep()
    except (rospy.ROSInterruptException, SystemExit, KeyboardInterrupt):
        sys.exit(0)

