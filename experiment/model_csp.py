import sys
sys.path.append('/data/sonnh8/TextDetection/DifferentiableBinarization')
from functools import wraps, reduce

import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Layer, MaxPooling2D, Lambda,
                                     ZeroPadding2D, UpSampling2D, Input)
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
import os

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * tf.math.sigmoid(inputs)

    def get_config(self):
        config = super(SiLU, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

class Focus(Layer):
    '''
    RegOrg module (?)
    '''
    def __init__(self, **kwargs):
        super(Focus, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] != None else input_shape[1],
         input_shape[2] // 2 if input_shape[2] != None else input_shape[2], input_shape[3] * 4)
    
#     def get_config(self):
#         config = super(Focus, self).get_config()
#         return config

    def call(self, x):
        return tf.concat(
            [x[...,  ::2,  ::2, :],
             x[..., 1::2,  ::2, :],
             x[...,  ::2, 1::2, :],
             x[..., 1::2, 1::2, :]],
             axis=-1
        )
#------------------------------------------------------#
#   DarknetConv2D
#   If set strides = 2 then do your own padding
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02),
     'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   DarknetConv2D + BatchNormalization + SiLU
#---------------------------------------------------#
def DarknetConv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(momentum = 0.97, epsilon = 0.001, name = kwargs['name'] + '.bn'),
        SiLU())

#---------------------------------------------------#
#   SPP: stacking after 3 max pooling is used
#---------------------------------------------------#
def SPPBottleneck(x, out_channels, weight_decay=5e-4, name = ""):
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    return x

def Bottleneck(x, out_channels, shortcut=True, weight_decay=5e-4, name = ""):
    y = compose(
            DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1'),
            DarknetConv2D_BN_SiLU(out_channels, (3, 3), weight_decay=weight_decay, name = name + '.conv2'))(x)
    if shortcut:
        y = Add()([x, y])
    return y

def CSPLayer(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)  # hidden channels
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))
    
    route = Concatenate()([x_1, x_2])
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name = name + '.conv3')(route)

def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, weight_decay=5e-4, name = ""):
    #----------------------------------------------------------------#
    #   reduce height and width by stride-2 convolution
    #----------------------------------------------------------------#

    # 320, 320, 64 => 160, 160, 128
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = name + '.0')(x)
    if last:
        x = SPPBottleneck(x, num_filters, weight_decay=weight_decay, name = name + '.1')
    return CSPLayer(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, weight_decay=weight_decay, name = name + '.1' if not last else name + '.2')

#---------------------------------------------------#
#   CSPdarkne main part
#   input shape: HxWx3 = 640x640x3
#   output three feature layers with stride: 1/8, 1/16, 1/32
#---------------------------------------------------#
def darknet_body(x, dep_mul, wid_mul, weight_decay=5e-4):
    base_channels   = int(wid_mul * 64)  # 64
    base_depth      = max(round(dep_mul * 3), 1)  # 3
    # 640, 640, 3 => 320, 320, 12
    x = Focus()(x)
    # 320, 320, 12 => 320, 320, 64
    x = DarknetConv2D_BN_SiLU(base_channels, (3, 3), weight_decay=weight_decay, name = 'backbone.backbone.stem.conv')(x)
    # 320, 320, 64 => 160, 160, 128
    x = resblock_body(x, base_channels * 2, base_depth, weight_decay=weight_decay, name = 'backbone.backbone.dark2')
    # 160, 160, 128 => 80, 80, 256
    x = resblock_body(x, base_channels * 4, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark3')
    feat1 = x
    # 80, 80, 256 => 40, 40, 512
    x = resblock_body(x, base_channels * 8, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark4')
    feat2 = x
    # 40, 40, 512 => 20, 20, 1024
    x = resblock_body(x, base_channels * 16, base_depth, shortcut=False, last=True, weight_decay=weight_decay, name = 'backbone.backbone.dark5')
    feat3 = x
    return feat1,feat2,feat3

#---------------------------------------------------#
#   PANet
#---------------------------------------------------#
def PANet(features, depth, width, in_channels=[256,  512, 1024], weight_decay=5e-4):
    feat1, feat2, feat3 = features
    P5          = DarknetConv2D_BN_SiLU(int(in_channels[1] * width), (1, 1), weight_decay=weight_decay,
                                        name = 'backbone.lateral_conv0')(feat3)  
    P5_upsample = UpSampling2D()(P5)  # 512/16
    P5_upsample = Concatenate(axis = -1)([P5_upsample, feat2])  # 512->1024/16
    P5_upsample = CSPLayer(P5_upsample, int(in_channels[1] * width), round(3 * depth), shortcut = False,
                           weight_decay=weight_decay, name = 'backbone.C3_p4')  # 1024->512/16

    P4          = DarknetConv2D_BN_SiLU(int(in_channels[0] * width), (1, 1), weight_decay=weight_decay,
                                        name = 'backbone.reduce_conv1')(P5_upsample)  # 512->256/16
    P4_upsample = UpSampling2D()(P4)  # 256/8
    P4_upsample = Concatenate(axis = -1)([P4_upsample, feat1])  # 256->512/8
    P3_out      = CSPLayer(P4_upsample, int(in_channels[0] * width), round(3 * depth), shortcut = False,
                           weight_decay=weight_decay, name = 'backbone.C3_p3')  # 1024->512/16

    P3_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P3_out)
    P3_downsample   = DarknetConv2D_BN_SiLU(int(in_channels[0] * width), (3, 3), strides = (2, 2), weight_decay=weight_decay,
                                            name = 'backbone.bu_conv2')(P3_downsample)  # 256->256/16
    P3_downsample   = Concatenate(axis = -1)([P3_downsample, P4])  # 256->512/16
    P4_out          = CSPLayer(P3_downsample, int(in_channels[1] * width), round(3 * depth), shortcut = False,
                               weight_decay=weight_decay, name = 'backbone.C3_n3')  # 1024->512/16

    P4_downsample   = ZeroPadding2D(((1, 0),(1, 0)))(P4_out)
    P4_downsample   = DarknetConv2D_BN_SiLU(int(in_channels[1] * width), (3, 3), strides = (2, 2), weight_decay=weight_decay,
                                            name = 'backbone.bu_conv1')(P4_downsample)  # 256->256/16
    P4_downsample   = Concatenate(axis = -1)([P4_downsample, P5])  # 512->1024/32
    P5_out          = CSPLayer(P4_downsample, int(in_channels[2] * width), round(3 * depth), shortcut = False,
                               weight_decay=weight_decay, name = 'backbone.C3_n4')  # 1024->512/16

    fpn_outs    = [P3_out, P4_out, P5_out]
    return fpn_outs

root = os.path.abspath(os.path.dirname(__file__))

class CSPDarknet:
    def __init__(self, input_shape, model_type='tiny', weight_decay=5e-4):
        assert model_type in ['tiny', 's', 'm', 'l', 'x'], 'Model type not found'
        self.image_input = Input(shape=input_shape)
        self.depth_dict  = {'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        self.width_dict  = {'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        self.model_weight_paths = {
            'tiny': 'csp_weights/yolox_tiny.h5',
            's': 'csp_weights/yolox_s.h5',
            'm': 'csp_weights/yolox_m.h5',
            'l': 'csp_weights/yolox_l.h5',
            'x': 'csp_weights/yolox_x.h5'
        }
        self.model_type  = model_type
        self.weight_decay = weight_decay
        
    
    def build(self):
        depth, width    = self.depth_dict[self.model_type], self.width_dict[self.model_type]
        in_channels     = [256, 512, 1024]

        #---------------------------------------------------#
        #   Input 640, 640, 3
        #   feat1 80, 80, 256
        #   feat2 40, 40, 512
        #   feat3 20, 20, 1024
        #---------------------------------------------------#
        feat1, feat2, feat3 = darknet_body(self.image_input, depth, width, weight_decay=self.weight_decay)
        fpn_outs = PANet([feat1, feat2, feat3], depth, width, in_channels, self.weight_decay)
        model = tf.keras.models.Model(inputs=self.image_input, outputs=fpn_outs)
        model.load_weights(os.path.join(root, self.model_weight_paths[self.model_type]), by_name=True)
        return model

from experiment.asf import ScaleFeatureSelection 

class ImplicitA:
    def __init__(self, channel, mean=0., std=.02):
        self.channel = channel
        self.mean = mean
        self.std = std
        initializer = tf.random_normal_initializer(mean=mean, stddev=std)
        self.implicit = tf.Variable(initializer(shape=(1, 1, 1, channel)), dtype=tf.float32, trainable=True)

    def __call__(self, x):
        return self.implicit + x

class ImplicitM:
    def __init__(self, channel, mean=1., std=.02):
        self.channel = channel
        self.mean = mean
        self.std = std
        initializer = tf.random_normal_initializer(mean=mean, stddev=std)
        self.implicit = tf.Variable(initializer(shape=(1, 1, 1, channel)), dtype=tf.float32, trainable=True)

    def __call__(self, x):
        return self.implicit * x

class CSPDBNet:
    def __init__(self, hparams):
        self.k = hparams['k']
        self.image_input = tf.keras.layers.Input(shape=(None, None, 3))
        if 'csp_type' in hparams.keys():
            model_type = hparams['csp_type']
        else:
            model_type = 'tiny'
        self.backbone = CSPDarknet(input_shape=(None, None, 3), model_type=model_type).build()
         
        self.use_asf = hparams['use_asf']
        self.asf = ScaleFeatureSelection(256, 64, 3, 'scale_channel_spatial')
    
    def make_model(self):
        P3, P4, P5 = self.backbone(self.image_input)
        kernel_reg = tf.keras.regularizers.l2(5e-4)
        P3 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, use_bias=False, padding='same',
                        kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(P3)
        P4 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=4, use_bias=False, padding='same',
                        kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(P4)
        P5 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=8, use_bias=False, padding='same',
                        kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(P5)
        
        # fuse = layers.Concatenate()([P3, P4, P5])
        # if self.use_asf:
        #     fuse = self.asf(fuse, [P3, P4, P5])
        P3 = ImplicitM(P3.shape[-1])(P3)
        P4 = ImplicitM(P4.shape[-1])(P4)
        P5 = ImplicitM(P5.shape[-1])(P5)
        fuse = P3 + P4 + P5
        fuse = ImplicitA(fuse.shape[-1])(fuse)
            
        # probability map
        p = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
        p = layers.BatchNormalization()(p)
#         p = layers.ReLU()(p)
        p = tf.keras.activations.swish(p)
        p = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2),
                                   kernel_initializer='he_normal', use_bias=False, kernel_regularizer=kernel_reg)(p)
        p = layers.BatchNormalization()(p)
#         p = layers.ReLU()(p)
        p = tf.keras.activations.swish(p)
        p = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2),
                                   kernel_initializer='he_normal', kernel_regularizer=kernel_reg)(p)
        p = tf.math.sigmoid(p, name='probality')
   
        # threshold map
        t = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
        t = layers.BatchNormalization()(t)
#         t = layers.ReLU()(t)
        t = tf.keras.activations.swish(t)
        t = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2),
                                   kernel_initializer='he_normal', use_bias=False, kernel_regularizer=kernel_reg)(t)
        t = layers.BatchNormalization()(t)
#         t = layers.ReLU()(t)
        t = tf.keras.activations.swish(t)
        t = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2),
                                   kernel_initializer='he_normal',kernel_regularizer=kernel_reg)(t)
        t = tf.math.sigmoid(t, name='threshold')

        # approximate binary map
        b_hat = 1 / (1 + tf.exp(-self.k * (p-t)))
        out = layers.Concatenate()([p, t, b_hat])
        return tf.keras.models.Model(inputs=self.image_input, outputs=out)
    

if __name__ == '__main__':

    ## Test back bone CSP ##
    '''
    tiny: 3.9M || s: 7M || m: 21M || l: 46.6M || x: 87.3M
    '''
    # model = CSPDarknet((640, 640, 3), 'x').build()
    # model.summary()
    # outs = model.output
    # for out in outs:
    #     print(out.shape)

    ## -------------- ##
    ## Test intire model ##

    import numpy as np
    from time import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#     for i in range(5):
#         p = np.random.random((1, 10, 10, 1))
#         t = np.random.random((1, 10, 10, 1))
#         s = time()
#         lbd = layers.Lambda(lambda x: 1 / (1 + tf.exp(-50 * (x[0] - x[1]))))([p, t])
#         dr = 1 / (1 + tf.exp(-50 * (p-t)))
# #         print(time() - t, t - s)
#         print(tf.reduce_sum(lbd))
#         print(tf.reduce_sum(dr))
    def test_net():
        hparams = {'k':50, 'csp_type':'tiny', 'use_asf':True}
        model = CSPDBNet(hparams).make_model()
        model.summary()
        #test time
        # out = model.get_layer('tf.math.sigmoid').output
        out = model.output[..., 0]
        model_infer = tf.keras.models.Model(inputs=model.input, outputs=out)
        data = np.random.random((5, 640, 1280, 3))
        model(tf.expand_dims(data[0], 0))
        for d in data:
            s = time()
            res = model_infer(tf.expand_dims(d, 0), training=False)
            print(time()-s)
            print(res.shape)
    def test_spatial_attn():
        attn = ScaleChannelSpatialAttention()
        x = layers.Input(shape=(20, 20, 64), batch_size=5)
        res = attn(x)
        print(res)
    
    def test_asf():
        asf = ASF()
        feature_list = [layers.Input(shape=(20, 20, 64), batch_size=5) for i in range(4)]
        x = asf(feature_list)
        print(x)
        
    test_net()
 



        

