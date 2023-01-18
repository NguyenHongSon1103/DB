import cv2
import math
import numpy as np
import pyclipper
import tensorflow as tf
from shapely.geometry import Polygon
import os
from time import time
import scipy.io as sio
from xml.etree import ElementTree as ET
import pickle as pkl
import json
import sys
sys.path.append('/data/sonnh8/TextDetection/DifferentiableBinarization')
from augmenter import RandomRotate, RandomCrop, RandomFlip


def parse_xml(xml):
    root = ET.parse(xml).getroot()
    objs = root.findall('object')
    boxes, ymins, obj_names = [], [], []
    for obj in objs:
        obj_name = obj.find('name').text
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        ymins.append(ymin)
        boxes.append([[xmin, ymin], [xmax, ymin],
                      [xmax, ymax], [xmin, ymax]])
        obj_names.append(obj_name)
    indices = np.argsort(ymins)
    boxes = [boxes[i] for i in indices]
    return np.array(boxes, dtype=np.float)

def parse_json(jpath):
    with open(jpath, 'r') as f:
        lb = json.load(f)
    boxes = [s['points'] for s in lb['shapes'] if len(s['points']) == 4]
    return np.array(boxes, dtype=np.float)

def read_anno_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        raws = f.read().split('\n')
        raws = [raw.split(',')[:8] for raw in raws if raw != '']
    polys = [[float(r) for r in raw] for raw in raws]
    return np.array(polys).reshape((-1, 4, 2))

def load_synthtext(path):
    mat = sio.loadmat(path)
    image_names = mat['imnames'][0]
    bboxes = mat['wordBB'][0]
    texts = mat['txt'][0]
    def preprocess_box(boxes):
        if len(np.shape(boxes)) == 2:
            boxes = np.array([boxes])
            boxes = np.transpose(boxes, (1, 2, 0))
        boxes = np.transpose(boxes, (2, 1, 0))
        return boxes
    bboxes = [preprocess_box(boxes) for boxes in bboxes]
    image_names = [name[0] for name in image_names]
    return image_names, bboxes 
                                            
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

def resize_2(image_short_side, image, polys):
    h, w = image.shape[:2]
    if h < w:
        h_new = image_short_side
        w_new = int(w / h * h_new / 32) * 32
        
    else:
        w_new = image_short_side
        h_new = int(h / w * w_new / 32) * 32
    resized_img = cv2.resize(image, (w_new, h_new))
    new_polys = []
    for p in polys:
        p = np.array(p)
        p[:, 0] = np.clip(p[:, 0] / w * w_new, 0, w_new)
        p[:, 1] = np.clip(p[:, 1] / h * h_new, 0, h_new)
        new_polys.append(p)
    return resized_img, new_polys

def resize_random(image, polys, w_range=(400, 800)):
    h, w = image.shape[:2]
    w_new = np.random.randint(w_range[0], w_range[1])
    w_new = (w_new // 32) * 32
    h_new = int(h / w * w_new / 32) * 32
    resized_img = cv2.resize(image, (w_new, h_new))
    new_polys = []
    for p in polys:
        p = np.array(p)
        p[:, 0] = np.clip(p[:, 0] / w * w_new, 0, w_new)
        p[:, 1] = np.clip(p[:, 1] / h * h_new, 0, h_new)
        new_polys.append(p)
    return resized_img, new_polys

class Dataset(tf.keras.utils.Sequence):
    def __init__(self, data, hparams, mode='train'):
        self.shrink_ratio = hparams['shrink_ratio']
        self.thresh_min=0.3
        self.thresh_max=0.7
        self.min_box_length = hparams['min_box_length'] # 8 with SynthText dataset
        self.batch_size = 1 #hparams['batch_size']
        self.im_size = hparams['input_size']
        self.mode = mode
        self.data = data
        self.w_range = hparams['w_resize_range']
        self.gray_scale_training = hparams['gray_scale_training']
        self.augmenters = [RandomRotate(limit=10)]
#                            RandomCrop(size=(self.im_size, self.im_size))]
#                            RandomFlip()]
           
    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch = self.data[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_images, batch_gts, batch_thresh_maps, batch_bhats = [], [], [], []
        for b in batch:
            img = cv2.imread(b['im_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            polygons = [b['boxes']]
            imgs = [img]    
#             if self.mode == 'train':
#                 for aug in self.augmenters:
#                     im, pl = aug(img, b['boxes'])
#                     imgs.append(im)
#                     polygons.append(pl)
#                 apply data augmentation: RandomRotation(-10, 10), RandomCropping, RandomFlipping         
            for img, single_poly in zip(imgs, polygons):
                single_poly = np.array([poly for poly in single_poly if Polygon(poly).is_valid])
                if len(single_poly) == 0:
                    continue
                if self.mode == 'val':
                    img, polys = resize_2(self.im_size, img, single_poly)
                else:
                    img, polys = resize_random(img, single_poly, self.w_range)
                h, w = img.shape[:2]
                prob_map = np.zeros((h, w), dtype=np.float32)
                thresh_map = np.zeros((h, w), dtype=np.float32)
                mask = np.zeros((h, w), dtype=np.float32)
                bhat_map = np.ones((h, w), dtype=np.float32)
                
                for poly in polys:
#                     try:
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
#                     except:
#                         print('come here')
#                         pass
                thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
                ##Convert image to gray scale
                if self.gray_scale_training:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img = np.stack([gray, gray, gray], -1)
                mean = [103.939, 116.779, 123.68]
                img = img.astype(np.float32)
                img[..., 0] -= mean[0]
                img[..., 1] -= mean[1]
                img[..., 2] -= mean[2]
#                 img = img / 255.0
                batch_images.append(img)
                batch_gts.append(prob_map)
                batch_thresh_maps.append(thresh_map)
                batch_bhats.append(bhat_map)
#                 if prob_map.sum() == 0:
#                     print(b['im_path'])
        return np.array(batch_images), np.stack([np.array(batch_gts), np.array(batch_thresh_maps), np.array(batch_bhats)], -1)
        
    def on_epoch_end(self):
        np.random.shuffle(self.data)

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
    
def get_synthtext_generator(gt_path, hparams):
    s = time()
    image_names, word_bboxes = load_synthtext(gt_path)
    print('Print: Training SynthText dataset \n Loaded %d annotations in %.3f(s)'%(len(image_names), time()-s))
    split_idx = int(len(image_names) * 0.01)
    val_data = [{'im_path':os.path.join('/data/sonnh8/TextDetection/SynthText', name), 'boxes': word_boxes}
                for name, word_boxes in zip(image_names[:split_idx], word_bboxes[:split_idx])]
    train_data = [{'im_path':os.path.join('/data/sonnh8/TextDetection/SynthText', name), 'boxes': word_boxes}
                for name, word_boxes in zip(image_names[split_idx:], word_bboxes[split_idx:])]
    train_gen = Dataset(train_data, hparams, mode='train')
    val_gen = Dataset(val_data, hparams, mode='val')
    return train_gen, val_gen

def is_valid_image(name):
    return ('.jpg' in name) or ('.png' in name) or ('.jpeg' in name)

def get_invoice_generator(hparams):
    print('Training: Invoice ')
    def get_files(path, lb_ext='.xml'):
        names = []
        labels = []
        for iv_type in os.listdir(path):
            print(iv_type)
            sub_f = os.path.join(path, iv_type)
            im_names = os.listdir(sub_f)
            im_names = [name for name in im_names if is_valid_image(name)]
            names += [os.path.join(sub_f, name) for name in im_names]
            labels += [os.path.join(sub_f, '.'.join(name.split('.')[:-1]) + lb_ext) for name in im_names]
        return names, labels

    train_names, train_labels = get_files(hparams['train_data'], lb_ext='.json')
    train_data = [{'im_path':name, 'boxes':parse_json(lb)} for name, lb in zip(train_names, train_labels)]
    train_gen = Dataset(train_data, hparams, mode='train')
    val_names, val_labels = get_files(hparams['val_data'], lb_ext='.json')
    val_data = [{'im_path':name, 'boxes':parse_json(lb)} for name, lb in zip(val_names, val_labels)]
    val_gen = Dataset(val_data, hparams, mode='val')
    return train_gen, val_gen

def get_synth_doc_generator(hparams):
#     def get_files(path):
#         names, labels = [], []
#         for name in os.listdir(path):
#             if 'json' in name:
#                 continue
#             fp = os.path.join(path, name)
#             lp = os.path.join(path, '.'.join(name.split('.')[:-1]) + '.json')
#             if not os.path.exists(lp):
#                 continue
#             names.append(fp)
#             labels.append(lp)
#         return names, labels
    def get_files(path):
        names = []
        labels = []
        for iv_type in os.listdir(path):
            sub_f = os.path.join(path, iv_type)
            im_names = os.listdir(sub_f)
            im_names = [name for name in im_names if is_valid_image(name)]
            names += [os.path.join(sub_f, name) for name in im_names]
            labels += [os.path.join(sub_f, '.'.join(name.split('.')[:-1]) + '.json') for name in im_names]
        return names, labels
    
    train_names, train_labels = get_files(hparams['train_data'])
    train_data = [{'im_path':name, 'boxes':parse_json(lb)} for name, lb in zip(train_names, train_labels)]
    train_gen = Dataset(train_data, hparams, mode='train')
    val_names, val_labels = get_files(hparams['val_data'])
    val_data = [{'im_path':name, 'boxes':parse_json(lb)} for name, lb in zip(val_names, val_labels)]
    val_gen = Dataset(val_data, hparams, mode='val')
    return train_gen, val_gen 
            

if __name__ == '__main__':
    from config.synth_doc_hparams import hparams
#     train_gen, val_gen = get_invoice_generator()
    train_gen, val_gen = get_synth_doc_generator(hparams)
    print(len(train_gen), len(val_gen))
    for i in range(len(train_gen)):
        print(train_gen.data[i]['im_path'])
        s = time()
        x = train_gen[i]
        print(x[0].shape, x[1].shape, time()-s)
        print(x[1][..., 0].sum(), x[1][..., 1].sum())
    
        


