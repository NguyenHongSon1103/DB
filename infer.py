import tensorflow as tf
import math
import cv2
import os
import glob
import numpy as np
from shapely.geometry import Polygon
import pyclipper
from time import time
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.ERROR)
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def resize_image(image, image_short_side):
    h, ư = image.shape[:2]
    if h < w:
        h_new = image_short_side
        w_new = int(w / h * h_new / 32) * 32
    else:
        w_new = image_short_side
        h_new = int(h / w * w_new / 32) * 32
    resized_img = cv2.resize(image, (w_new, h_new))
    return resized_img    

def scale_polys(size, h, w, polys):
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    new_anns = []
    for poly in polys:
        poly = np.array(poly).astype(np.float32)
        poly /= scale
        new_anns.append(poly.astype('int32'))
    return new_anns

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    subject = [tuple(l) for l in box]
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
    height, width = bitmap.shape[:2]
    boxes, scores = [], []

    contours, _ = cv2.findContours((bitmap * 255.0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape((-1, 2)))
        if box_thresh > score:
            continue
        
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=1.5)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        if len(box) == 0: continue
        box, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue
        box = np.array(box)
        box[:, 0] = np.clip(box[:, 0] / width * dest_width, 0, dest_width)
        box[:, 1] = np.clip(box[:, 1] / height * dest_height, 0, dest_height)
        boxes.append(box.astype('int32'))
        scores.append(score)
    if max_candidates == -1:
        return boxes, scores
    idxs = np.argsort(scores)
    scores = [scores[i] for i in idxs[:max_candidates]]
    boxes = [boxes[i] for i in idxs[:max_candidates]]
        
    return boxes, scores

def prob2heatmap(prob_map, rgb=True):
    heatmap = cv2.applyColorMap((prob_map*255).astype('uint8'), cv2.COLORMAP_JET)
    if not rgb:
        heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    return heatmap

def to_json(coords, size, image_path, json_path, labels=None):
    h, w = size
    json_dicts = {'shapes':[], 'imagePath':image_path, 'imageData':None, 'imageHeight':h, 'imageWidth':w}
    if labels is None:
        for coord in coords:
    #         print(coord)
            json_dicts['shapes'].append({'label':'text', 'points':coord.tolist(), 'shape_type':'polygon', 'flags':{}})
    else:
        for coord, label in zip(coords, labels):
            json_dicts['shapes'].append({'label':label, 'points':coord.tolist(), 'shape_type':'polygon', 'flags':{}})

    with open(json_path, 'w') as f:
        json.dump(json_dicts, f)


model_infer = tf.saved_model.load('path-to-model')
data = 'path-to-data'
save = False
view = True
save_path = 'path-to-save-data'
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
for num, name in enumerate(os.listdir(data)):
    if 'xml' in name:
        continue
    if 'json' in name or '_srinkmap' in name or '.DS_Store' in name:
        continue
    sp_json = os.path.join(save_path, '.'.join(name.split('.')[:-1])+'.json')
    if 'xml' in name or 'txt' in name: continue
    start = time()
    fp = os.path.join(data, name)
    image = cv2.imread(fp)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w = image.shape[:2]e
    src_image = image.copy()
    image = resize_image(image, image_short_side=960)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.stack([gray, gray, gray], -1)

    print(num, src_image.shape, image.shape)
    image_input = np.expand_dims(image, axis=0)
    p = model_infer(image_input)[0].numpy()
    print('inference time: ', time()-start)
    bitmap = p > 0.2
    boxes, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=0.2, max_candidates=-1)
    for i, box in enumerate(boxes):
        src_image = cv2.drawContours(src_image, [box], -1, (0, 255, 0), 2)
    heatmap = prob2heatmap(bitmap, True)
    if save:
        to_json(boxes, (h, w), name, os.path.join(save_path, '.'.join(name.split('.')[:-1])+'.json'))

    print('total time: ', time()-start)
    if view:
        plt.figure(figsize=(10, 10))
        plt.imshow(src_image)
        plt.figure(figsize=(10, 10))
        plt.imshow(heatmap)
        plt.show()
        input()
    