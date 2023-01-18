import datetime
import os
import tensorflow as tf
# from experiment.generator_freesize import get_synthtext_generator, get_invoice_generator, get_synth_doc_generator
from generator import get_synth_doc_generator
from model import DBNet
from experiment.fast_model import TextNasA2
from config.synth_doc_hparams import hparams
import tensorflow_addons as tfa
from losses import *
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = hparams['gpu']
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

root = os.getcwd()
save_model_dir = os.path.join(root, hparams['model_dir'])
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)
## Get dataset ##
train_gen, val_gen = get_synth_doc_generator(hparams)
print('DATASET: \n TRAINING DATA: %d \n VALIDATION DATA: %d'%(len(train_gen), len(val_gen)))

## Define teacher and student model ##
with open('trained_models/synth_doc_crop_20221228/model.json') as f:
    teacher_model = tf.keras.models.model_from_json(f.read())
    teacher_model.load_weights('trained_models/synth_doc_crop_20221228/weights_31_0.7169.h5')
    out = teacher_model.output[..., :1]
    teacher_model = tf.keras.models.Model(inputs=teacher_model.input, outputs=out)

student_model = TextNasA2().make_model()
print(teacher_model.output.shape, student_model.output_shape)

with open(os.path.join(save_model_dir, 'model.json'), 'w') as f:
    f.write(student_model.to_json())
if hparams['resume']:
    print('USING PRETRAINED WEIGHTS: ', hparams['pretrained_weights'])
    student_model.load_weights(hparams['pretrained_weights'], by_name=True)

## Define loss, optimizers, callbacks
if hparams['steps_per_epoch'] is None:
    max_iters = hparams['epochs']*len(train_gen)
else:
    max_iters = hparams['epochs']*hparams['steps_per_epoch']
lr = tf.keras.optimizers.schedules.PolynomialDecay(
    hparams['base_lr'], max_iters, hparams['end_lr'], power=0.9)

opt = tf.keras.optimizers.Adam(learning_rate = lr)
# opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
model_dir = os.path.join(save_model_dir, 'weights_{epoch:02d}_{val_loss:.4f}.h5')

## Training: 
train_log_dir = os.path.join(save_model_dir, 'logs', 'train')
val_log_dir = os.path.join(save_model_dir, 'logs', 'val')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

def loss(student_out, teacher_out, labels):
    p_true = labels[..., :1]
    ## KD loss
    student_teacher_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(teacher_out, student_out))

    student_label_loss, _ = balanced_crossentropy_loss(p_true, student_out, 3.0)

    student_teacher_loss, student_label_loss = 0.8*student_teacher_loss, 0.2*student_label_loss
    loss_dict = dict(
        student_teacher_loss = student_teacher_loss,
        student_label_loss   = student_label_loss,
        total_loss           = student_teacher_loss + student_label_loss
    )
    return loss_dict

def val_loss(student_out, labels):
    p_true = labels[..., :1]
    student_label_loss, _ = balanced_crossentropy_loss(p_true, student_out, 3.0)
    return student_label_loss

def train_step(student_model, teacher_model, iter_data):
    
    images, labels = iter_data

    p_teacher = teacher_model(images, training=False)

    with tf.GradientTape() as tape:
        p_student = student_model(images, training=True)

        loss_dict = loss(p_student, p_teacher, labels)

        grads = tape.gradient(loss_dict['total_loss'], student_model.trainable_variables)
        opt.apply_gradients(zip(grads, student_model.trainable_variables))

    return loss_dict

current_val_loss = 1e5
KEYS = ['student_teacher_loss', 'student_label_loss', 'total_loss']
train_loss_dict = {key:tf.keras.metrics.Mean() for key in KEYS}
num_steps = len(train_gen)
for epoch in range(hparams['epochs']):
    print('-'*10, 'Epoch: %d/%d'%(epoch, hparams['epochs']), '-'*10)
    print('%10s\t%10s\t%10s'%tuple(KEYS)+ 'lr')
    pbar = tqdm(range(num_steps))#, total=train_len, position=0, leave=True)
    for num_iter in pbar:
        iter_data = train_gen[num_iter]
        
        loss_dict = train_step(student_model, teacher_model, iter_data)
        
        for key in loss_dict:
            train_loss_dict[key](loss_dict[key])
        
        pbar.set_description(
                            '%8.4f   %8.4f   %8.4f'%tuple(train_loss_dict[key].result() for key in KEYS)+
                            '   %8.4fe-3'%(opt._decayed_lr('float32').numpy()*1000))
        ##Log each 20 step
        if num_iter % 100 == 0:
            with train_summary_writer.as_default():
                for key in KEYS:
                    tf.summary.scalar(key, train_loss_dict[key].result(), step=num_iter)
        
    #Epoch end
    train_gen.on_epoch_end()
        
    ## Validation phase
    ## RUN validation each 1k iter
    print(' ==== EVALUATION ===== ')
    val_loss_avg = tf.keras.metrics.Mean()
    for n_batch, d in enumerate(val_gen):
        images, labels = d
        val_preds = student_model(images, training=False)
        _loss = val_loss(val_preds, labels)
        val_loss_avg(_loss)

    val_loss_res = val_loss_avg.result()
    print(f"Epoch: [{epoch}]\t", '%8.4f'%val_loss_res)
    
    model_dir = os.path.join(save_model_dir, 'weights_%d_%.4f.h5'%(epoch, val_loss_res))
    student_model.save_weights(model_dir)

    with val_summary_writer.as_default():
        tf.summary.scalar(key, val_loss_res, step=epoch)

    print(' ==== END EVAL ====')

