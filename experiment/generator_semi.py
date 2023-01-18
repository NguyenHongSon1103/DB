import cv2
import math
import numpy as np
import pyclipper
import tensorflow as tf
from shapely.geometry import Polygon
import os
from time import time
import pickle as pkl
import json
import sys
sys.path.append('/data/sonnh8/TextDetection/DifferentiableBinarization')
from augmenter import RandomRotate, RandomCrop, RandomFlip, RandomBrightnessContrast

def parse_json(jpath):
    with open(jpath, 'r') as f:
        lb = json.load(f)
    boxes = [s['points'] for s in lb['shapes'] if len(s['points']) == 4]
    return np.array(boxes, dtype=np.float)

def resize(size, image, polys):
    '''
    Output shape: (size, size, 3), pad for keep aspect ratio
    '''
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

def resize_2(image_short_side, image, polys):
    '''
    Output shape: (short_side, h, 3), pad for keep aspect ratio
    '''
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

class Dataset(tf.keras.utils.Sequence):
    def __init__(self, data, hparams, mode='train'):
        self.shrink_ratio = hparams['shrink_ratio']
        self.thresh_min=0.3
        self.thresh_max=0.7
        self.min_box_length = 1 # 8 with SynthText dataset
        self.batch_size = hparams['batch_size']
        self.im_size = hparams['input_size']
        self.mode = mode
        self.data = data
        
#         self.augmenters = [RandomRotate(limit=10),
#                            RandomCrop(size=(self.im_size, self.im_size))]
#                            RandomFlip()]
#         self.ratio = len(self.pseudo_labeled_data) // len(self.labeled_data)
#         self.ratio = 1 if self.ratio == 0 else self.ratio
                                           
    def __len__(self):
        return len(self.data) // self.batch_size
    
    def get_data_label(self, img, single_poly, im_size):
        h, w = im_size[:2]
        prob_map = np.zeros((h, w), dtype=np.float32)
        thresh_map = np.zeros((h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.float32)
        bhat_map = np.ones((h, w), dtype=np.float32)

        for poly in single_poly:
#             try:
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
#             except:
#                         print('come here')
#                 pass
        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
        return prob_map, thresh_map, bhat_map
        
    def __getitem__(self, idx):
        batch = self.data[idx*self.batch_size : (idx+1)*self.batch_size]
        images, labels = [], []
        
        for b in batch:
            img = cv2.imread(b['im_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(b['boxes'])
#             if self.mode == 'train':
#                 for aug in self.augmenters:
#                     im, pl = aug(img, b['boxes'])
#                     images.append(im)
#                     labels.append(pl)
                #apply data augmentation: RandomRotation(-10, 10), RandomCropping, RandomFlipping  
                          
        batch_images, batch_gts, batch_thresh_maps, batch_bhats = [], [], [], []
        for img, single_poly in zip(images, labels):
            single_poly = np.array([poly for poly in single_poly if Polygon(poly).is_valid])
            if len(single_poly) == 0:
                continue
            img, single_poly = resize(self.im_size, img, single_poly)
#             if self.mode == 'val':
#                 img, single_poly = resize_2(self.im_size, img, single_poly)
#             else:
#                 img, single_poly = resize_random(img, single_poly, (600, 800))
                
            prob_map, thresh_map, bhat_map = self.get_data_label(img, single_poly, img.shape[:2])
            img = img.astype('float32') - [103.939, 116.779, 123.68]
            batch_images.append(img)
            batch_gts.append(prob_map)
            batch_thresh_maps.append(thresh_map)
            batch_bhats.append(bhat_map)
#             print(prob_map.sum())# == 0:
#                 print(b['im_path'])
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
                          
class SemiDataset(tf.keras.utils.Sequence):
    def __init__(self, data, hparams):
        self.batch_size = hparams['batch_size_semi']
        self.im_size = hparams['input_size']
        self.data = data
        self.visual_augmenters = [RandomBrightnessContrast(0.5)]
                                           
    def __len__(self):
        return len(self.data) // self.batch_size
        
    def __getitem__(self, idx):
        batch = self.data[idx*self.batch_size : (idx+1)*self.batch_size]
        images = []
        
        for b in batch:
            img = cv2.imread(b['im_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for augmenter in self.visual_augmenters:
                img = augmenter(img)
            images.append(img)
                          
        batch_images = []
        for img in images:
            img, _ = resize(self.im_size, img, [])
           
            img = img.astype('float32') - [103.939, 116.779, 123.68]
            batch_images.append(img)
        return np.array(batch_images)
                          
    def on_epoch_end(self):
        np.random.shuffle(self.data)
                             
def is_valid_image(name):
    return ('.jpg' in name) or ('.png' in name) or ('.jpeg' in name)

def get_generator(hparams):
    
    def get_files(path, lb_ext='.json'):
        names = []
        labels = []
        for iv_type in os.listdir(path):
            sub_f = os.path.join(path, iv_type)
            im_names = os.listdir(sub_f)
            im_names = [name for name in im_names if is_valid_image(name)]
            len_names = len(im_names)
            names += [os.path.join(sub_f, name) for name in im_names]
            labels += [os.path.join(sub_f, '.'.join(name.split('.')[:-1]) + lb_ext) for name in im_names]
        return names, labels
    
    def get_semi_files(path):
        im_names = os.listdir(path)
        names = [os.path.join(path, name) for name in im_names if is_valid_image(name)]
        return names

    train_names, train_labels = get_files(hparams['train_data'], lb_ext='.json')
    train_data = [{'im_path':name, 'boxes':parse_json(lb)} for name, lb in zip(train_names, train_labels)]
    
    train_unlabel_names = get_semi_files(hparams['train_unlabel_data'])
#     train_unlabel_names, _ = get_files(hparams['train_unlabel_data'])
#     print(len(train_pseudo_names), len(train_pseudo_labels), len(probmaps_pseudo))
    train_unlabel_data = [{'im_path':name} for name in train_unlabel_names]
    ## Shuffle unlabel data before fit in
    np.random.shuffle(train_unlabel_data)
    
    train_gen = Dataset(train_data, hparams, mode='train')
    train_unlabel_gen = SemiDataset(train_unlabel_data, hparams)
    val_names, val_labels = get_files(hparams['val_data'], lb_ext='.json')
    val_data = [{'im_path':name, 'boxes':parse_json(lb)} for name, lb in zip(val_names, val_labels)]
    val_gen = Dataset(val_data, hparams, mode='val')
    
    return train_gen, val_gen, train_unlabel_gen
        
if __name__ == '__main__':
    from time import time
    from config.semi_doc_hparams import hparams
    train_gen, val_gen, train_unlabel_gen = get_generator(hparams)
    print(len(train_gen), len(val_gen), len(train_unlabel_gen))
    for i in range(len(train_gen)%10):
        s = time()
        res = train_gen[i]
#         print(res[0].shape, res[1])
        print(i, time()-s )
#         assert False
        
    
    
        


