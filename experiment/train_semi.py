import datetime
import os
import tensorflow as tf
from experiment.generator_semi import get_generator
from model import DBNet
from config.semi_hparams import hparams
from experiment.losses import semi_loss
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

root = os.getcwd()
save_model_dir = os.path.join(root, hparams['model_dir'])
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)
    
## Get dataset ##
train_gen, val_gen, train_unlabeled_gen = get_generator(hparams)
print('DATASET: \n TRAINING DATA: %d \n VALIDATION DATA: %d \
      \n UNLABELED DATA: %d'%(len(train_gen), len(val_gen), len(train_unlabeled_gen)))

#*** GET MODEL ***#
#Get teacher model
# with open('trained_models/synth_doc_2/model.json', 'r') as f:
#     teacher_model = tf.keras.models.model_from_json(f.read())
teacher_model.load_weights('trained_models/synth_doc_2/weights_22_0.7480.h5') 
      
#Get student model
student_model = DBNet().make_model()
with open(os.path.join(save_model_dir, 'model_train.json'), 'w') as f:
    f.write(student_model.to_json())
if hparams['resume']:
    print('USING PRETRAINED WEIGHTS: ', hparams['pretrained_weights'])
    student_model.load_weights(hparams['pretrained_weights'])
      
#*** END GET MODEL ***#

num_steps = hparams['num_steps']
lr = tf.keras.optimizers.schedules.PolynomialDecay(
    hparams['base_lr'], num_steps, hparams['end_lr'], power=0.9
)
# opt = tf.keras.optimizers.Adam(learning_rate = lr)
opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

train_log_dir = os.path.join(save_model_dir, 'logs', 'train')
val_log_dir = os.path.join(save_model_dir, 'logs', 'val')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

def train_step(student_model, teacher_model,
               data_labeled, images_unlabeled,
               loss_fn):
    
    images, labels = data_labeled

    with tf.GradientTape() as tape:
        preds = student_model(images, training=True)
        student_unlabel_preds = student_model(images_unlabeled, training=True)
        teacher_unlabel_preds = teacher_model(images_unlabeled, training=False)
        loss_dict, total_loss = loss_fn(labels, preds, student_unlabel_preds, teacher_unlabel_preds)
        grads = tape.gradient(total_loss, student_model.trainable_variables)
        opt.apply_gradients(zip(grads, student_model.trainable_variables))
    return loss_dict
        
current_val_loss = 1e5
KEYS = ['total_loss', 'pmap_loss', 'tmap_loss', 'bhmap_loss', 'unsup_loss']
train_loss_dict = {key:tf.keras.metrics.Mean() for key in KEYS}

train_gen_length, train_unlabel_gen_length = len(train_gen), len(train_unlabeled_gen)
for num_iter in num_steps:
    data_labeled = train_gen[num_iter%train_gen_length]
    images_unlabeled = train_unlabeled_gen[num_iter%train_unlabel_gen_length]
    
    loss_dict = train_step(student_model, teacher_model,
                          data_labeled, images_unlabeled, semi_loss)
    
    t = None
    for key in loss_dict:
        train_loss_dict[key](loss_dict[key])
        
    ##Log each 20 step
    if num_iter % 20 == 0:
        t = t if t is not None else time.time()
        elapsed = time.time() - t
        str_loss = ', \t'.join(['%s: %.4f'%(key, loss_dict[key]) for key in KEYS])
        print(f"Step: %d, \t time : %.2f , \t %s"%(n_batch, elapsed, str_loss))          
        t = time.time()
        with train_summary_writer.as_default():
            for key in KEYS:
                tf.summary.scalar(key, train_loss_dict[key].result(), step=num_iter)
    
    ## shuffle data if loaded all batch in generator
    if num_iter >= train_gen_length:
        train_gen.on_epoch_end()
    if num_iter >= train_unlabel_gen_length:
        train_unlabeled_gen.on_epoch_end()
    
    ## Validation phase
    ## RUN validation each 1k iter
    if num_iter % 1e3 == 0:
        val_loss_dict = {key:tf.keras.metrics.Mean() for key in KEYS}
        for n_batch, d in enumerate(val_gen):
            images, labels = d
            val_preds = student_model(images, training=False)
            loss_dict, _ = semi_loss(labels, val_preds)  
            for key in loss_dict:
                val_loss_dict[key](loss_dict[key])
        
        print(' ==== EVALUATION ===== ') 
        print(f"Step: [{num_iter}]\t", {k:'%3f'%val_loss_dict[k] for k in val_loss_dict})
        print(' ==== END EVAL ====')
        
        if current_val_loss > val_loss_dict['total_loss']:
            model_dir = os.path.join(save_model_dir, 'weights_%.4f.h5'%(val_loss_dict['total_loss']))
            student_model.save_weights(model_dir)
            current_val_loss = val_loss_dict['total_loss']

        with val_summary_writer.as_default():
            for key in KEYS:
                tf.summary.scalar(key, val_loss_dict[key].result(), step=num_iter)

             