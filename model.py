import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from experiment.asf import ScaleFeatureSelection 

class DBNet:
    def __init__(self, hparams):
        self.k = hparams['k']
        self.image_input = layers.Input(shape=(None, None, 3))
#         self.backbone = tf.keras.applications.EfficientNetB0(input_tensor=self.image_input, include_top=False)
#         self.out_layers = ['block2b_activation', 'block3b_activation', 'block5a_activation', 'block7a_activation']
        self.backbone = tf.keras.applications.ResNet50(input_tensor=self.image_input, include_top=False)
#         self.out_layers = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
        self.out_layers = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block3_out', 'conv5_block1_out']
        if hparams['fine_tune']:
            self.backbone.trainable = False
         
        self.use_asf = hparams['use_asf']
        self.asf = ScaleFeatureSelection(256, 64, 4, 'scale_channel_spatial')
    
    def make_model(self):
        C2, C3, C4, C5 = [self.backbone.get_layer(l).output for l in self.out_layers]
        in2 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2) # 160x160
        in3 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3) # 80x40
        in4 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4) # 40x40
        in5 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5) # 20x20
        # 1 / 32 * 8 = 1 / 4
        P5 = layers.UpSampling2D(size=(8, 8))(
            layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5)) # 160x160
        # 1 / 16 * 4 = 1 / 4
        out4 = layers.Add()([in4, layers.UpSampling2D(size=(2, 2))(in5)]) # 40x40
        P4 = layers.UpSampling2D(size=(4, 4))(
            layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4)) # 160x160
        # 1 / 8 * 2 = 1 / 4
        out3 = layers.Add()([in3, layers.UpSampling2D(size=(2, 2))(out4)]) # 80x80
        P3 = layers.UpSampling2D(size=(2, 2))(
            layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3)) # 160x160
        # 1 / 4
        P2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(
            layers.Add()([in2, layers.UpSampling2D(size=(2, 2))(out3)])) # 160x160
        # (b, /4, /4, 256)
        fuse = layers.Concatenate()([P2, P3, P4, P5])
        if self.use_asf:
            fuse = self.asf(fuse, [P2, P3, P4, P5])
            
        kernel_reg = tf.keras.regularizers.l2(5e-4)
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
        return models.Model(inputs=self.image_input, outputs=out)
    

if __name__ == '__main__':
    import numpy as np
    from time import time
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
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
        hparams = {'k':50, 'fine_tune':False, 'use_asf':True}
        model = DBNet(hparams).make_model()
        model.summary()
        #test time
        out = model.output[..., 0]
        model_infer = models.Model(inputs=model.input, outputs=out)
        data = np.random.random((5, 640, 1280, 3))
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
 
