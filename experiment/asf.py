import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
    
class ScaleChannelAttention:
    def __init__(self, out_planes, num_features, init_weight=True):     
        self.avgpool = tfa.layers.AdaptiveAveragePooling2D((1, 1))
        self.fc1 = layers.Conv2D(out_planes, kernel_size=1, padding='same', use_bias=False)
        self.fc2 = layers.Conv2D(num_features, kernel_size=1, padding='same', use_bias=False)
       
    def __call__(self, x):
        '''
        x.shape = bs, W, H, C
        '''
        global_x = self.avgpool(x)
        global_x = self.fc1(global_x)
        global_x = layers.ReLU()(layers.BatchNormalization()(global_x))
        global_x = self.fc2(global_x)
        global_x = tf.nn.softmax(global_x) #bs, W, H, 4
        return global_x
   
class ScaleChannelSpatialAttention:
    def __init__(self, in_planes, out_planes, num_features):     
        self.channel_wise = tf.keras.Sequential(layers=[
            tfa.layers.AdaptiveAveragePooling2D((1, 1)),
            layers.Conv2D(out_planes, kernel_size=1, padding='same', kernel_initializer='he_normal', use_bias=False),
            layers.ReLU(),
            layers.Conv2D(in_planes, kernel_size=1, padding='same', kernel_initializer='he_normal', use_bias=False),
        ])
        
        self.spatial_wise = tf.keras.Sequential(layers=[
            layers.Conv2D(1, kernel_size=3, padding='same', use_bias=False),
            layers.ReLU(),
            layers.Conv2D(1, kernel_size=1, padding='same', use_bias=False),
            #thiếu sigmoid ở đây
        ])
        
        self.attention_wise = tf.keras.Sequential(layers=[
            layers.Conv2D(in_planes, kernel_size=1, padding='same', kernel_initializer='he_normal', use_bias=False),
            #thiếu sigmoid ở đây
        ])
    def __call__(self, x):
        '''
        x.shape = NxHxWx256
        '''
        global_x = self.channel_wise(x) #bs, 1, 1, 256
        global_x = tf.nn.sigmoid(global_x) + x  #bs, 1, 1, 256
        x = tf.reduce_mean(global_x, 1, keepdims=True) 
#         print(x.shape)
        sw = self.spatial_wise(x)
        global_x = tf.nn.sigmoid(x) + global_x #bs, W, H, 4
        global_x = self.attention_wise(global_x)
        global_x = tf.nn.sigmoid(global_x) #bs, W, H, 4
        return global_x

#in_channels = [64, 128, 256, 512]
#inner_channels = 256
class ScaleFeatureSelection:
    def __init__(self, in_channels, inter_channels , out_features_num=4, attention_type='scale_spatial'):
        self.in_channels=in_channels
        self.inter_channels = inter_channels 
        self.out_features_num = out_features_num
        self.conv = layers.Conv2D(inter_channels, kernel_size=3, padding='same')
        self.type = attention_type
        if self.type == 'scale_channel_spatial':
            self.enhanced_attention = ScaleChannelSpatialAttention(inter_channels, inter_channels // 4, out_features_num)
        elif self.type == 'scale_channel':
            self.enhanced_attention = ScaleChannelAttention(inter_channels, inter_channels//2, out_features_num)
    
    def __call__(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        assert len(features_list) == self.out_features_num
        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i+1] * features_list[i])
        return tf.concat(x, -1)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import numpy as np
    #Input size: 640x640x3
    P5 = np.random.random((2, 160, 160, 64)) #1/16
    P4 = np.random.random((2, 160, 160, 64)) #1/8
    P3 = np.random.random((2, 160, 160, 64)) #1/4
    P2 = np.random.random((2, 160, 160, 64))
    fuse = np.concatenate([P5, P4, P3, P2], 1)
    asf = ScaleFeatureSelection(256, 64, 4, 'scale_channel_spatial')
    res = asf(fuse, [P5, P4, P3, P2])
    print(res.shape)