import cv2
import math
import numpy as np
import pyclipper
import tensorflow as tf
from shapely.geometry import Polygon
import os
from time import time
from xml.etree import ElementTree as ET
import pickle as pkl
import json
from tqdm import tqdm
from augmenter import RandomRotate, RandomCrop, RandomFlip

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

class Dataset(tf.keras.utils.Sequence):
    def __init__(self, data, hparams, mode='train'):
        self.shrink_ratio = hparams['shrink_ratio']
        self.thresh_min=0.3
        self.thresh_max=0.7
        self.min_box_length = hparams['min_box_length'] # 8 with SynthText dataset
        self.batch_size = hparams['batch_size']
        self.im_size = hparams['short_edge']
        self.mode = mode
        self.data = data
        self.gray_scale_training = hparams['gray_scale_training']

        self.augmenters = [RandomCrop(size=(self.im_size, self.im_size))]
#             RandomRotate(limit=10),
                           
#                            RandomFlip()]
           
    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        batch = self.data[idx*self.batch_size : (idx+1)*self.batch_size]
        batch_images, batch_gts, batch_thresh_maps, batch_bhats = [], [], [], []
        
        #Get all images and polygons in batch
        images, polygons = [], []
        for b in batch:
            img = cv2.imread(b['fp'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            polygons.append(parse_json(b['lp']))
            images.append(img)  
            # if self.mode == 'train':
            #     pass
            #     for aug in self.augmenters:
            #         im, pl = aug(img, b['boxes'])
            #         images.append(im)
            #         polygons.append(pl)
                #apply data augmentation: RandomRotation(-10, 10), RandomCropping, RandomFlipping 
        # Resize batch to the same shape
                
        for img, single_poly in zip(images, polygons):
            single_poly = np.array([poly for poly in single_poly if Polygon(poly).is_valid])
            if len(single_poly) == 0:
                continue
            img, polys = resize(self.im_size, img, single_poly)
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
#                         print('come here')
                    pass
            thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min
            #Process img
            if self.gray_scale_training:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.stack([gray, gray, gray], -1)
            mean = [103.939, 116.779, 123.68]
            img = img.astype(np.float32)
            img[..., 0] -= mean[0]
            img[..., 1] -= mean[1]
            img[..., 2] -= mean[2]
            
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

def is_valid_image(name):
    return ('.jpg' in name) or ('.png' in name) or ('.jpeg' in name)

def get_generator(hparams):
    def get_files(path):
        names, labels = [], []
        filenames = [name for name in os.listdir(path) if 'json' not in name]
        for name in tqdm(filenames, desc='Loading annotaions: '):
            fp = os.path.join(path, name)
            lp = os.path.join(path, fp[:-3]+'json')
            if not os.path.exists(lp):
                continue
            names.append(fp)
            labels.append(lp)
        return names, labels
    
    train_names, train_labels = get_files(hparams['train_data'])
    train_data = [{'fp':fp, 'lp':lp} for fp, lp in zip(train_names, train_labels)]
    train_gen = Dataset(train_data, hparams, mode='train')
    
    val_names, val_labels = get_files(hparams['val_data'])
    val_data = [{'fp':fp, 'lp':lp} for fp, lp in zip(val_names, val_labels)]
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
    
        


