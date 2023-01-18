import numpy as np
import cv2
from PIL import Image, ImageEnhance
import albumentations as A

class BaseAugmenter:
    def __init__(self, augmenter, size=None):
        self.size = size
        self.augmenter = augmenter
    
    def __call__(self, image, boxes):
        assert len(boxes.shape) == 3
        keypoints = np.copy(boxes.reshape((-1, 2)))
        trans = self.augmenter(image=image.copy(), keypoints=keypoints)
        aug_img = trans['image']
        aug_boxes = np.reshape(trans['keypoints'], (-1, 4, 2))
        size = self.size if self.size is not None else image.shape[:2]
        aug_boxes = self.validate_boxes(aug_boxes, size)
        return aug_img, aug_boxes

    def validate_boxes(self, boxes, size):
        size_h, size_w = size
        new_boxes = []
        for box in boxes:
            if np.sum(box[:, 0] > size_w) > 0:
                continue
            if np.sum(box[:, 1] > size_h) > 0:
                continue
            new_boxes.append(box)
        return np.array(new_boxes, dtype=np.int32)

class RandomCrop:
    def __init__(self, size):
        self.size_h, self.size_w = size
        scale = (0.08, 1.0)
        ratio = (0.75, 1.333333333333)
        augmenter = A.Compose([
            A.RandomResizedCrop(self.size_h, self.size_w,
            scale, ratio)],
            keypoint_params=A.KeypointParams(format='xy',
            remove_invisible=False))
        self.base_augmenter = BaseAugmenter(augmenter, size)
    
    def __call__(self, image, boxes):
        return self.base_augmenter(image, boxes)

class RandomRotate:
    def __init__(self, limit=10):
        augmenter = A.Compose([
            A.Rotate(limit, border_mode=cv2.BORDER_CONSTANT,
            value=[0, 0, 0], p=1.0)],
            keypoint_params=A.KeypointParams(format='xy',
            remove_invisible=False))
        self.base_augmenter = BaseAugmenter(augmenter, None)
    def __call__(self, image, boxes):
        return self.base_augmenter(image, boxes)

class RandomFlip:
    def __init__(self):
        augmenter = A.Compose([
            A.HorizontalFlip(p=1.0)],
            keypoint_params=A.KeypointParams(format='xy',
            remove_invisible=False))
        self.base_augmenter = BaseAugmenter(augmenter, None)
    def __call__(self, image, boxes):
        return self.base_augmenter(image, boxes)
    
class RandomBrightnessContrast:
    def __init__(self, p, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)):
        self.augmenter = A.Compose([
            A.RandomBrightnessContrast(p=p,
                                       brightness_limit=brightness_limit,
                                       contrast_limit=contrast_limit,
                                       brightness_by_max=True)])
    def __call__(self, image):
        trans = self.augmenter(image=image)
        return trans['image']
