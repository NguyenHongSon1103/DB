import tensorflow as tf
import math
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import os
from hparams import hparams
from time import time
import pickle as pkl
import json
from augmenter import RandomRotate, RandomCrop, RandomFlip
# from generator import get_invoice_generator

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def parse_json(jpath):
    with open(jpath, 'r') as f:
        lb = json.load(f)
    boxes = [s['points'] for s in lb['shapes'] if len(s['points']) == 4]
    return np.array(boxes, dtype=np.float)
                                            
def compute_distance(xs, ys, point_1,  point_2):
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / (square_distance))

    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    return result

def get_offset(shrink_ratio, poly, is_thresh_map=True):
    polygon = Polygon(poly)
    distance = polygon.area * (1 - np.power(shrink_ratio, 2)) / polygon.length
    subject = [tuple(l) for l in poly]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    if not is_thresh_map:
        shrinked = padding.Execute(-distance)
    else:
        shrinked = np.array(padding.Execute(distance)[0])
    return np.array(shrinked), distance

def resize(size, image, polys):
    h, w, c = image.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    padimg = np.zeros((size, size, c), image.dtype)
    padimg[:h, :w] = cv2.resize(image, (w, h))
    new_anns = []
    for poly in polys:
        poly = np.array(poly).astype(np.float32)
        poly *= scale
        new_anns.append(poly)
    return padimg, new_anns

class LabelGen:
    def __init__(self):
        self.shrink_ratio = hparams['shrink_ratio']
        self.thresh_min=0.3
        self.thresh_max=0.7
        self.min_box_length = 3 # 8 with SynthText dataset
        self.im_size = hparams['input_size']

    def __call__(self, polys, im_shape):

        prob_map = np.zeros((self.im_size, self.im_size), dtype=np.float32)
        thresh_map = np.zeros((self.im_size, self.im_size), dtype=np.float32)
        mask = np.zeros((self.im_size, self.im_size), dtype=np.float32)
        bhat_map = np.ones((self.im_size, self.im_size), dtype=np.float32)

        for poly in polys:
            try:
                height = max(poly[:, 1]) - min(poly[:, 1])
                width = max(poly[:, 0]) - min(poly[:, 0])
                polygon = Polygon(poly)
                # generate gt and mask
                if polygon.area < 1 or min(height, width) < self.min_box_length:
                    cv2.fillPoly(bhat_map, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                else:
                    shrinked, _ = get_offset(0.4, poly, False)
                    if len(shrinked) == 0:
                        cv2.fillPoly(bhat_map, poly.astype(np.int32)[np.newaxis, :, :], 0)
                        continue
                    else:
                        shrinked = np.array(shrinked[0]).reshape(-1, 2)
                        if shrinked.shape[0] > 2 and Polygon(shrinked).is_valid:
                            cv2.fillPoly(prob_map, [shrinked.astype(np.int32)], 1)
                        else:
                            cv2.fillPoly(bhat_map, poly.astype(np.int32)[np.newaxis, :, :], 0)
                            continue
                thresh_map, _ = self.draw_border_map(poly, thresh_map, mask)
            except:
                pass
        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
        return np.stack([prob_map, thresh_map, bhat_map], -1)
        
    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        padded_polygon, distance = get_offset(self.shrink_ratio, polygon, True)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymin+1, #ymax_valid-ymax+height
                xmin_valid-xmin:xmax_valid-xmin+1], #xmax_valid-xmax+width
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
        
        return canvas, mask
    
def is_valid_image(name):
    return ('.jpg' in name) or ('.png' in name) or ('.jpeg' in name)

if __name__ == '__main__':
    path = '/data/sonnh8/TextDetection/Receipt_dataset
    for in_type in os.listdir(path):
        sub_f = os.path.join(path, in_type)
        for name in os.listdir(sub_f):
            if not is_valid_image(name):
                continue
            fp = os.path.join(sub_f, name)
            jp = '.'.join(fp.split('.')[:-1]) + '.json'
            img = cv2.imread(fp)
            
    

polygons = np.array([poly for poly in single_poly if Polygon(poly).is_valid])
if len(single_poly) == 0:
    continue
def get_tfrecord(filename, generator):
    writer = tf.data.experimental.TFRecordWriter(filename)
    for (image, label) in generator:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image),
            'label': _bytes_feature(label)
        }))
        writer.write(tf_example.SerializeToString())
    writer.close()

train_gen, val_gen = get_invoice_generator()
# train_data = tf.data.Dataset.from_generator(train_gen)
get_tfrecord('val_semi_data.record', val_gen)

