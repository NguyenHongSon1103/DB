import tensorflow as tf
import tensorflow.keras.layers as L

def conv2d_block(inp, filters, kernel_size, stride, padding, use_bn=False):
    x = L.Conv2D(filters, kernel_size, stride, padding)(inp)
    x = L.ReLU()(x)
    if use_bn:
        x = L.BatchNormalization()(x)
    return x

class TextNasA2:
    def __init__(self):
        self.kernel_size =  7
        self.inp = L.Input((None, None, 3))

    def make_model(self):
        x = conv2d_block(self.inp, 64, (3, 3), 2, 'same')
        # x = L.ZeroPadding2D(((1, 0), (1, 0)))(x)
        x = conv2d_block(x, 64, (3, 3), 1, 'same')

        ## block 1 ##
        x = conv2d_block(x, 64, (3, 3), 2, 'same')
        x = conv2d_block(x, 64, (3, 1), 1, 'same')
        x = conv2d_block(x, 64, (3, 3), 1, 'same')
        x = conv2d_block(x, 64, (3, 1), 1, 'same')
        x = conv2d_block(x, 64, (3, 3), 1, 'same')
        x = conv2d_block(x, 64, (3, 3), 1, 'same')
        x = conv2d_block(x, 64, (1, 3), 1, 'same')
        x = conv2d_block(x, 64, (3, 3), 1, 'same')
        out_1 = conv2d_block(x, 64, (3, 3), 1, 'same', True)

        ## block 2 ##
        x = conv2d_block(out_1, 128, (3, 3), 2, 'same')
        x = conv2d_block(x, 128, (1, 3), 1, 'same')
        x = conv2d_block(x, 128, (3, 3), 1, 'same')
        x = conv2d_block(x, 128, (3, 1), 1, 'same')
        # x = conv2d_block(x, 128, (3, 3), 1, 'same')
        # x = conv2d_block(x, 128, (3, 3), 1, 'same')
        # x = conv2d_block(x, 128, (3, 1), 1, 'same')
        x = conv2d_block(x, 128, (3, 1), 1, 'same')
        x = conv2d_block(x, 128, (3, 3), 1, 'same')
        out_2 = conv2d_block(x, 128, (3, 3), 1, 'same', True)

        ## block 3 ##
        x = conv2d_block(out_2, 256, (3, 3), 2, 'same')
        x = conv2d_block(x, 256, (3, 3), 1, 'same')
        # x = conv2d_block(x, 256, (3, 3), 1, 'same')
        x = conv2d_block(x, 256, (1, 3), 1, 'same')
        # x = conv2d_block(x, 256, (3, 3), 1, 'same')
        x = conv2d_block(x, 256, (3, 1), 1, 'same')
        x = conv2d_block(x, 256, (3, 3), 1, 'same')
        out_3 = conv2d_block(x, 256, (3, 1), 1, 'same', True)

        ## block 4 ##
        x = conv2d_block(out_3, 512, (3, 3), 2, 'same')
        # x = conv2d_block(x, 512, (1, 3), 1, 'same')
        # x = conv2d_block(x, 512, (3, 1), 1, 'same')
        x = conv2d_block(x, 512, (3, 1), 1, 'same')
        out_4 = conv2d_block(x, 512, (1, 3), 1, 'same', True)

        ## detection head ##
        out_1 = L.Conv2D(128, 3, 1, 'same')(out_1)
        out_2 = L.Conv2D(128, 3, 1, 'same')(out_2)
        out_2 = L.UpSampling2D(2, interpolation='bilinear')(out_2)
        out_3 = L.Conv2D(128, 3, 1, 'same')(out_3)
        out_3 = L.UpSampling2D(4, interpolation='bilinear')(out_3)
        out_4 = L.Conv2D(128, 3, 1, 'same')(out_4)
        out_4 = L.UpSampling2D(8, interpolation='bilinear')(out_4)
        F = L.Concatenate()([out_1, out_2, out_3, out_4])
        F = L.Conv2D(128, 3, 1, 'same')(F)
        F = L.UpSampling2D(4, interpolation='bilinear')(F)
        F = L.Conv2D(128, 3, 1, 'same')(F)

        output = L.Conv2D(1, 3, 1, 'same')(F)
        output = tf.nn.sigmoid(output)
        # args: input, kernel size, stride, padding
       

        return tf.keras.Model(inputs=self.inp, outputs=output)

if __name__ == '__main__':
    from time import time
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    model = TextNasA2().make_model()
    model.summary()
    
    data = np.random.random((5, 640, 1280, 3))
    for d in data:
        s = time()
        res = model(tf.expand_dims(d, 0), training=False)
        print(res.shape)
        print(time()-s)
