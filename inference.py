import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class p2d:
    def __init__(self,xywh):
        self.x, self.y, self.w, self.h = xywh
        print(xywh)
class y3():
    def __init__(self, weights):
        # parameters
        self.augment = False
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        # model
        device = select_device('')
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(640, s=stride) # 640
        names = model.module.names if hasattr(model,'module') else model.names
        if half:
            model.half()
        if device.type != 'cpu':
            model(torch.zeros(1,3,imgsz,imgsz).to(device).type_as(next(model.parameters())))
        self.img_size = imgsz
        self.stride = stride
        self.names = names
        self.half = half
        self.device = device
        self.model = model

    @torch.no_grad()
    def inference(self, frame):
        """inference 1 frame"""
        t0 = time.time()
        img = letterbox(frame, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
            max_det=self.max_det)
        t2 = time_synchronized()

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                # Print results
                s = ''
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    # Plot results
                    if True:
                        c = int(cls)
                        label = f'{self.names[c]} {conf:.2f}'
                        plot_one_box(xyxy, frame, label=label, color=colors(c, True), line_thickness=1)
                    # Make points
                    if True:
                        gn = torch.tensor(frame.shape)[[1,0,1,0]]
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) # label format
                        p2d(xywh)
                        # with open(txt_path + '.txt', 'a') as f:
                            # f.write(('%g ' * len(line)).rstrip() % line + '\n')

        # Print time (inference + NMS)
        print(f'{s}Done. ({t2 - t1:.3f}s)')

        return frame

if __name__ == "__main__":
    frame = cv2.imread('data/test/49.jpg')
    cv2.imshow('sa',y3('best0.pt').inference(frame))
    cv2.waitKey(0)
    plt.imshow(frame)
    plt.show()