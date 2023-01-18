import datetime
import os
import tensorflow as tf
from experiment.generator_semi import get_generator
from model import DBNet
from config.semi_doc_hparams import hparams
from experiment.semi_losses import semi_loss, db_loss
import time
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = hparams['gpu']
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

root = os.getcwd()
save_model_dir = os.path.join(root, hparams['model_dir'])
if not os.path.exists(save_model_dir):
    os.mkdir(save_model_dir)

def ema_update(model, model_ema, beta=0.99):
    """
    Performs a model update by using exponential moving average (EMA) 
    of first model's weights to update second model's weights follow by:
    
    model_ema.weights = beta * model_ema.weights + (1 - beta) * model.weights
    
    :param model: original, gradient descent trained model.
    :param model_ema: clone of original model, that will get updated using EMA.
    :param beta: EMA update weight, determines what portion of model_ema weights should be kept (default=0.9999).
    """
    # for each model layer index
    for i in range(len(model.layers)):
        updating_weights = model.layers[i].get_weights() # original model's weights
        ema_old_weights = model_ema.layers[i].get_weights() # ema model's weights
        ema_new_weights = [] # ema model's update weights
        if len(updating_weights) != len(ema_old_weights):
            # weight list length mismatch between model's weights and ema model's weights
            print("Different weight length")
            # copy ema weights directly from the model's weights
            ema_new_weights = updating_weights
        else:
            # for each weight tensor of original model's weights list
            for j in range(len(updating_weights)):
                n_weight = beta * ema_old_weights[j] + (1 - beta) * updating_weights[j]
                ema_new_weights.append(n_weight)
        # update weights
        model_ema.layers[i].set_weights(ema_new_weights)

## Get dataset ##
print('==== LOADING DATASET ====')
train_gen, val_gen, train_unlabeled_gen = get_generator(hparams)
print('DATASET: \n TRAINING DATA: %d \n VALIDATION DATA: %d \
      \n UNLABELED DATA: %d'%(len(train_gen), len(val_gen), len(train_unlabeled_gen)))
print('==== DONE ====')

#*** GET MODEL ***#
print('==== INITIALIZE FROM SCRATCH TEACHER / STUDENT MODEL ====')
gput, gpus = '/GPU:0', '/GPU:1'
#Get teacher model
with tf.device(gput):
    teacher_model = DBNet(hparams).make_model()
for l in teacher_model.layers:
    l.trainable = False
      
#Get student model
with tf.device(gpus):
    student_model = DBNet(hparams).make_model()
with open(os.path.join(save_model_dir, 'model.json'), 'w') as f:
    f.write(student_model.to_json())
    
print('==== DONE ====')
#*** END GET MODEL ***#

trained_steps = 0
if hparams['resume']:
    print('USING PRETRAINED WEIGHTS: ', hparams['student_pretrained_weights'])
    student_model.load_weights(hparams['student_pretrained_weights'])
    temp = hparams['student_pretrained_weights'].split('_')[-2]
    try:
        trained_steps = int(temp)
    except: pass

print('==== START TRAINING ====')
num_steps = hparams['num_steps']

lr = tf.keras.optimizers.schedules.PolynomialDecay(
    hparams['base_lr'], num_steps, hparams['end_lr'], power=0.9)

opt = tf.keras.optimizers.Adam(learning_rate = lr)
# opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

train_log_dir = os.path.join(save_model_dir, 'logs', 'train')
val_log_dir = os.path.join(save_model_dir, 'logs', 'val')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)

def train_step(student_model, teacher_model,
               data_labeled, images_u, step, total_step):
    
    images, labels = data_labeled
    #resize images_u to images for batching
#     h, w = tf.shape(images)[1], tf.shape(images)[2]
#     images_u = tf.image.resize(images_u, [h, w])
    sup_len = images.shape[0]
    images = tf.concat([images, images_u], 0)

    with tf.device(gput):
        T_u_preds = teacher_model(images_u, training=False)

    with tf.device(gpus):
        with tf.GradientTape() as tape:
            outs = student_model(images, training=True)
            S_l_preds, S_u_preds = outs[:sup_len], outs[sup_len:] 
    
            ## Caculate supervise loss
            alpha_0, epoch = 0.2, step//1000 + 1
            total_epoch = total_step // 1000 + 1
            # alpha_t = (1 - alpha_0) * (1-epoch/total_epoch)
            alpha_t = epoch/total_epoch
            loss, total_loss = semi_loss(labels, S_l_preds, S_u_preds, T_u_preds, alpha_t)

            grads = tape.gradient(total_loss, student_model.trainable_variables)
            opt.apply_gradients(zip(grads, student_model.trainable_variables))

    #Update teacher model with EMA
    ema_update(student_model, teacher_model, 0.999)
    return loss

current_val_loss = 1e5
KEYS = ['pmap_loss', 'tmap_loss', 'bhmap_loss', 'unsup_loss', 'total_loss']
train_loss_dict = {key:tf.keras.metrics.Mean() for key in KEYS}

train_gen_length, train_unlabel_gen_length = len(train_gen), len(train_unlabeled_gen)
t = None

# print('-'*10, 'Iter 1/%d'%num_steps, '-'*10)
print('%10s\t%10s\t%10s\t%10s\t%10s\t'%tuple(KEYS)+ '%10s'%'lr')
pbar = tqdm(range(trained_steps+1, num_steps))#, total=train_len, position=0, leave=True)
for num_iter in pbar:
    data_labeled = train_gen[num_iter%train_gen_length]
    images_u = train_unlabeled_gen[num_iter%train_unlabel_gen_length]
    
    loss_dict = train_step(student_model, teacher_model,
                          data_labeled, images_u, num_iter, num_steps)
    
    for key in loss_dict:
        train_loss_dict[key](loss_dict[key])
    
    pbar.set_description(
                         '%8.4f   %8.4f   %8.4f   %8.4f   %8.4f'%tuple(train_loss_dict[key].result() for key in KEYS)+
                         '   %8.5f'%opt._decayed_lr('float32').numpy())
    ##Log each 20 step
    if num_iter % 20 == 0:
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
    if num_iter > 0 and num_iter % 1000 == 0:
        print(' ==== EVALUATION ===== ')
        val_loss_dict = {key:tf.keras.metrics.Mean() for key in KEYS}
        for n_batch, d in enumerate(val_gen):
            images, labels = d
            val_preds = student_model(images, training=False)
            loss_dict = db_loss(labels, val_preds)
            for key in loss_dict:
                val_loss_dict[key](loss_dict[key])
         
        print(f"Step: [{num_iter}]\t", {k:'%3f'%val_loss_dict[k].result() for k in val_loss_dict})
        
        val_total_loss = val_loss_dict['total_loss'].result()
        model_dir = os.path.join(save_model_dir, 'weights_%d_%.4f.h5'%(num_iter, val_total_loss))
        student_model.save_weights(model_dir)

        with val_summary_writer.as_default():
            for key in KEYS:
                tf.summary.scalar(key, val_loss_dict[key].result(), step=num_iter)

        print(' ==== END EVAL ====')
