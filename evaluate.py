import os
import numpy as np
from shapely.geometry import Polygon
import json
import tensorflow as tf
import cv2
import pyclipper
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def resize_image(image, image_short_side):
    h, w = image.shape[:2]
    if h < w:
        h_new = image_short_side
        w_new = int(w / h * h_new / 32) * 32
    else:
        w_new = image_short_side
        h_new = int(h / w * w_new / 32) * 32
    resized_img = cv2.resize(image, (w_new, h_new))
    return resized_img

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

def infer(model, image, prob_thresh=0.2, box_thresh=0.3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    image = resize_image(image, image_short_side=640)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.stack([gray, gray, gray], -1)
    image_input = np.expand_dims(image, axis=0)
    p = model(image_input)[0].numpy()
    bitmap = p > prob_thresh
    boxes, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=box_thresh, max_candidates=-1)
#     print(bitmap.shape)
#     print(len(boxes[0]), len(boxes[1]))
    return np.array(boxes)

def parse_json(jpath):
    with open(jpath, 'r') as f:
        lb = json.load(f)
    boxes = [s['points'] for s in lb['shapes'] if len(s['points']) == 4]
    return np.array(boxes, dtype=np.float)

def bb_intersect_over_union(boxA, boxB, areaA, areaB):
    polyA, polyB = Polygon(boxA), Polygon(boxB)
    intersect = polyA.intersection(polyB).area
    union = areaA + areaB - intersect
    return intersect / union

def bb_iou_rect_mode(boxA, boxB):
    xminA, xmaxA = np.min(boxA[:, 0]), np.max(boxA[:, 0])
    yminA, ymaxA = np.min(boxA[:, 1]), np.max(boxA[:, 1])
    xminB, xmaxB = np.min(boxB[:, 0]), np.max(boxB[:, 0])
    yminB, ymaxB = np.min(boxB[:, 1]), np.max(boxB[:, 1])
    
    xA = max(xminA, xminB)
    yA = max(yminA, yminB)
    xB = min(xmaxA, xmaxB)
    yB = min(ymaxA, ymaxB)
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (xmaxA - xminA + 1) * (ymaxA - yminA + 1)
    boxBArea = (xmaxB - xminB + 1) * (ymaxB - yminB + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
    
def eval_single_image(gt_boxes, pred_boxes, iou_constraint=0.5):
    gt_boxes = [p for p in gt_boxes if Polygon(p).is_valid]
    pred_boxes = [p for p in pred_boxes if Polygon(p).is_valid]
    gt_boxes_len, pred_boxes_len = len(gt_boxes), len(pred_boxes)
    outputShape=[gt_boxes_len,pred_boxes_len]
    gtRecMat = np.zeros(gt_boxes_len, np.uint8)
    detRecMat = np.zeros(pred_boxes_len, np.uint8)
    iouMat = np.empty(outputShape)
    gt_areas = [Polygon(p).area for p in gt_boxes]
    pred_areas = [Polygon(p).area for p in pred_boxes]
    for gt_id in range(gt_boxes_len):
        for pr_id in range(pred_boxes_len):
            pG, pP = gt_boxes[gt_id], pred_boxes[pr_id]
            pG_area, pP_area = gt_areas[gt_id], pred_areas[pr_id]
            iouMat[gt_id, pr_id] = bb_intersect_over_union(pG, pP, pG_area, pP_area)
#             iouMat[gt_id, pr_id] = bb_iou_rect_mode(pG, pP)
    num_det_mat = 0
    for gt_id in range(gt_boxes_len):
        for pr_id in range(pred_boxes_len):
            if gtRecMat[gt_id] == 0 and detRecMat[pr_id] == 0:
                if iouMat[gt_id, pr_id] > iou_constraint:
                    gtRecMat[gt_id], detRecMat[pr_id] = 1, 1
                    num_det_mat += 1

    recall = float(num_det_mat) / gt_boxes_len if gt_boxes_len > 0 else 0
    precision = float(num_det_mat) / pred_boxes_len if pred_boxes_len > 0 else 0
    hmean = 2*recall*precision / (recall+precision) if (recall+precision) > 0 else 0
    res_dict = dict(
        precision = precision,
        recall = recall,
        hmean = hmean,
        num_det_mat = num_det_mat,
        gt_len = gt_boxes_len,
        pred_len = pred_boxes_len
    )
    return res_dict

if __name__ == '__main__':
    from time import time
    from argparse import ArgumentParser
    import logging
    logging.basicConfig(level=logging.ERROR)
    parser = ArgumentParser()
    parser.add_argument('-l', '--label_path', type=str, required=True)
    parser.add_argument('-i', '--image_path', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-it', '--iou_threshold', type=float, default=0.5)
    parser.add_argument('-pt', '--prob_threshold', type=float, default=0.2)
    parser.add_argument('-bt', '--box_threshold', type=float, default=0.3)
    parser.add_argument('--per_image', action='store_true')
    args = parser.parse_args()
    
    model = tf.saved_model.load(args.model)
    
    i, total_det_mat, total_gt_len, total_pred_len = 0, 0, 0, 0
    s = time()
    for name in os.listdir(args.image_path):
        if 'json' in name:
            continue
        fp = os.path.join(args.image_path, name)
        lp = os.path.join(args.label_path, '.'.join(name.split('.')[:-1])+'.json')
        if not os.path.exists(lp):
            print('label of %s can not found ==> continue'%name)
            continue
        i += 1
        gt_boxes = parse_json(lp)
        image = cv2.imread(fp)
        pred_boxes = infer(model, image.copy(), prob_thresh=args.prob_threshold,
                      box_thresh=args.box_threshold)
        res_dict = eval_single_image(gt_boxes, pred_boxes, iou_constraint=args.iou_threshold)
        if args.per_image:
            print('%s: %.3f\t%.3f\t%.3f'%(name, res_dict['precision'], res_dict['recall'],
                                          res_dict['hmean']))
        total_det_mat += res_dict['num_det_mat']
        total_gt_len += res_dict['gt_len']
        total_pred_len += res_dict['pred_len']
        P = total_det_mat / float(total_pred_len)
        R = total_det_mat / float(total_gt_len)
        F1= 2*R*P/(R+P)
    print('Evaluated on %d images with in %.3f(s), iou threshold: %f ==> Precision: %.4f\tRecall: %.4f\tF1: %.4f'%(i,time()-s, args.iou_threshold, P, R, F1))
        
        
                    
                    
    