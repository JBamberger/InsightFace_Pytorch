import torch
torch.cuda.current_device()
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


from mtcnn_pytorch.src.box_utils import nms, _preprocess
import math
from utils import load_facebank, draw_box_name, prepare_facebank
from Learner import face_learner
from mtcnn import MTCNN
from config import get_config
import numpy as np
from collections import OrderedDict
import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe, Value, Array




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_first_stage(image, net, scale, threshold):

    width, height = image.size
    sw, sh = math.ceil(width*scale), math.ceil(height*scale)
    img = image.resize((sw, sh), Image.BILINEAR)
    img = np.asarray(img, 'float32')

    img = torch.FloatTensor(_preprocess(img)).to(device)
    with torch.no_grad():
        output = net(img)
        probs = output[1].cpu().data.numpy()[0, 1, :, :]
        offsets = output[0].cpu().data.numpy()

        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        #keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes #[keep]


def _generate_bboxes(probs, offsets, scale, threshold):
    stride = 2
    cell_size = 12

    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    bounding_boxes = np.vstack([
        np.round((stride*inds[1] + 1.0)/scale),
        np.round((stride*inds[0] + 1.0)/scale),
        np.round((stride*inds[1] + 1.0 + cell_size)/scale),
        np.round((stride*inds[0] + 1.0 + cell_size)/scale),
        score, offsets
    ])

    return bounding_boxes.T


class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        weights = np.load('mtcnn_pytorch/src/weights/pnet.npy',
                          allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        # print(x.size())
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        # print(x.size())
        # print(a.size())
        # print(b.size())
        a = F.softmax(a, dim=-1)
        return b, a


if __name__ == '__main__':

    conf = get_config(False)

    pnet = PNet().to('cuda')
    pnet.eval()

    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    min_face_size = 20.0

    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            image = Image.fromarray(frame[...,::-1])
            # BUILD AN IMAGE PYRAMID
            width, height = image.size
            min_length = min(height, width)
            min_detection_size = 12
            factor = 0.707  # sqrt(0.5)
            scales = []  # scales for scaling the image
            m = min_detection_size/min_face_size
            min_length *= m

            factor_count = 0
            while min_length > min_detection_size:
                scales.append(m*factor**factor_count)
                min_length *= factor
                factor_count += 1

            # STAGE 1

            bounding_boxes = []  # it will be returned

            with torch.no_grad():
                # run P-Net on different scales
                for s in scales:
                    boxes = run_first_stage(
                        image, pnet, scale=s, threshold=0.7)
                    bounding_boxes.append(boxes)

                # collect boxes (and offsets, and scores) from different scales
                bounding_boxes = [i for i in bounding_boxes if i is not None]
                if (not bounding_boxes):
                    continue

                bounding_boxes = np.vstack(bounding_boxes)

                # keep = nms(bounding_boxes[:, 0:5], 0.7)
                bounding_boxes = bounding_boxes # [keep]
                bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])


            for i in range(bounding_boxes.shape[0]):
                bbox = bounding_boxes[i, :]
                frame = cv2.rectangle(frame,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),6)
            
            cv2.imshow('face Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
