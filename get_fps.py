import tensorflow as tf
import numpy as np
from time import time
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gpu', type=int, help='gpu for testing')
parser.add_argument('--savedmodel', type=str, help='path to model')

args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

dump = np.random.randint(256, size=(1, 640, 1280, 3)).astype('uint8')

model = tf.saved_model.load(args.savedmodel)

model(dump)

print('-'*5, 'Test with %s'%('cpu' if args.gpu == -1 else 'gpu'), '-'*5)

## ------ Test case 1 --------- ##
print('Test case 1: batch size = 1')
data = np.random.randint(256, size=(10, 640, 1280, 3)).astype('uint8')
total = 0
for d in data:
    s = time()
    model(tf.expand_dims(d, 0))
    t = time()
    runtime = t - s
    total += runtime
    print('Runtime: ', runtime)
print('Average: %.3f ==> FPS: %.3f'%(total/10, 10/total))

## ------ Test case 2 --------- ##
print('Test case 1: batch size = 4')
total = 0
dump = np.random.randint(256, size=(4, 640, 1280, 3)).astype('uint8')
model(dump)
for i in range(5):
    data = np.random.randint(256, size=(4, 640, 1280, 3)).astype('uint8')
    s = time()
    model(data)
    t = time()
    runtime = t - s
    total += runtime
    print('Runtime: ', runtime)
print('Average: %.3f ==> FPS: %.3f'%(total/20, 20/total))

'''
Report:
 Resnet50: 
     cpu: bs=1 ==> 0.96 FPS || bs=4 ==> 0.98 FPS
     gpu: bs=1 ==> 34.6 FPS || bs=4 ==> 115 FPS
 Student model: 
     cpu: bs=1 ==> 0.54 FPS || bs=4 ==> 0.59 FPS
     gpu: bs=1 ==> 28.7 FPS || bs=4 ==> 65.0 FPS
'''