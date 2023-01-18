import datetime
import os
import tensorflow as tf
# from experiment.generator_freesize import get_synthtext_generator, get_invoice_generator, get_synth_doc_generator
from generator import get_synth_doc_generator
from model import DBNet
# from experiment.model_csp import CSPDBNet
from config.synth_doc_hparams import hparams
# from config.hparams import hparams
from losses import db_loss
import tensorflow_addons as tfa

os.environ['CUDA_VISIBLE_DEVICES'] = hparams['gpu']
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

root = os.getcwd()
save_model_dir = os.path.join(root, hparams['model_dir'])
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)
## Get dataset ##
# train_gen, val_gen = get_synthtext_generator('/data/sonnh8/TextDetection/SynthText/gt.mat')
# train_gen, val_gen = get_invoice_generator()
train_gen, val_gen = get_synth_doc_generator(hparams)
print('DATASET: \n TRAINING DATA: %d \n VALIDATION DATA: %d'%(len(train_gen), len(val_gen)))

## Define model ##
model_train = DBNet(hparams).make_model()
# model_train = CSPDBNet(hparams).make_model()
model_train.summary()
with open(os.path.join(save_model_dir, 'model.json'), 'w') as f:
    f.write(model_train.to_json())
if hparams['resume']:
    print('USING PRETRAINED WEIGHTS: ', hparams['pretrained_weights'])
    model_train.load_weights(hparams['pretrained_weights'], by_name=True)

## Define loss, optimizers, callbacks
if hparams['steps_per_epoch'] is None:
    max_iters = hparams['epochs']*len(train_gen)
else:
    max_iters = hparams['epochs']*hparams['steps_per_epoch']
lr = tf.keras.optimizers.schedules.PolynomialDecay(
    hparams['base_lr'], max_iters, hparams['end_lr'], power=0.9)
# wd = tf.keras.optimizers.schedules.PolynomialDecay(
#     1e-5, max_iters, 1e-7, power=0.9
# )
# opt = tfa.optimizers.AdamW(
#     weight_decay = 1e-4, learning_rate = lr)
# lr = hparams['base_lr']
opt = tf.keras.optimizers.Adam(learning_rate = lr)
# opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
model_dir = os.path.join(save_model_dir, 'weights_{epoch:02d}_{val_loss:.4f}.h5')
ckpt = tf.keras.callbacks.ModelCheckpoint(model_dir, mode='min', monitor='val_loss',
                                         save_best_only=True, save_weights_only=True)
tfboard = tf.keras.callbacks.TensorBoard(os.path.join(save_model_dir, 'logs'))

## Training: 
model_train.compile(optimizer=opt, loss=db_loss)

model_train.fit( train_gen, epochs=hparams['epochs'],
                steps_per_epoch=hparams['steps_per_epoch'],
                callbacks=[ckpt, tfboard], validation_data=val_gen)

