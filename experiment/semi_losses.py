import tensorflow as tf
import numpy as np
# import tensorflow_addons as tfa

def balanced_crossentropy_loss(gt, pred, negative_ratio=3.):
    '''
    Args:
        pred: (b, h, w, 1)
        gt: (b, h, w, 1)
    Returns:
    
    '''
    positive_mask, negative_mask = gt[..., 0], 1-gt[..., 0]
    positive_count = tf.reduce_sum(positive_mask)
    negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])
    loss = tf.keras.losses.binary_crossentropy(gt, pred)
    positive_loss = loss * positive_mask
    negative_loss = loss * negative_mask
    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))

    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
            positive_count + negative_count + 1e-6)
    balanced_loss = balanced_loss
    return balanced_loss, loss

def dice_loss(bhat_true, bh_pred, mask, weights):
    """

    Args:
        bh_pred: (b, h, w, 1)
        bhat_true: (b, h, w, 1)
        mask: (b, h, w, 1)
        weights: (b, h, w)
    Returns: scalar

    """
    weights = (weights - tf.reduce_min(weights)) / (tf.reduce_max(weights) - tf.reduce_min(weights)) + 1.
    mask = bh_pred * weights
    intersection = tf.reduce_sum(bh_pred * bhat_true * mask)
    union = tf.reduce_sum(bh_pred * mask) + tf.reduce_sum(bhat_true * mask) + 1e-6
    loss = 1 - 2.0 * intersection / union
    return loss


def l1_loss(t_true, t_pred, mask):
    num_positive = tf.reduce_sum(mask)
    loss = tf.reduce_sum(tf.abs(t_pred - t_true) * mask) / num_positive
    return loss

def l1_loss_wo_mask(ytrue, ypred):
    loss = tf.reduce_mean(tf.abs(ytrue - ypred))
    return loss

def db_loss(ytrue, ypred):
    # L = Ls + alpha*Lb+ beta*Lt
    alpha, beta = 1.0, 10.0
    p_true, t_true, bhat_true = ytrue[..., :1], ytrue[..., 1], ytrue[..., 2]
    p_pred, t_pred, bhat_pred = ypred[..., :1], ypred[..., 1], ypred[..., 2] 
#     print(p_true, p_pred)
    t_mask = tf.cast(t_true > 0.3, dtype=tf.float32)
    loss_t = l1_loss(t_true, t_pred, t_mask)
    loss_p, dice_weights = balanced_crossentropy_loss(p_true, p_pred)
#     bce_loss_bhat = balanced_crossentropy_loss(bhat_true, bhat_pred)
    loss_bhat = dice_loss(bhat_true, bhat_pred, p_true, dice_weights)
    return {'tmap_loss':beta*loss_t, 'pmap_loss': alpha*loss_p,
            'bhmap_loss':loss_bhat,
            'total_loss':alpha*loss_p + beta*loss_t + loss_bhat }

def compute_supervise_loss(ypred, ytrue):
    return db_loss(ytrue, ypred)

def compute_unsupervise_loss(S_u_pred, T_u_pred, alpha_t):
    t_p, t_t, t_bhat = T_u_pred[..., :1], T_u_pred[..., 1], T_u_pred[..., 2]
    s_p, s_t, s_bhat = S_u_pred[..., :1], S_u_pred[..., 1], S_u_pred[..., 2] 

    # entropy = -tf.reduce_sum(t_p*tf.math.log(t_p+1e-10), -1)
    # gamma_t = np.percentile(entropy.numpy().flatten(), 100*(1-alpha_t))
    # labels_u = tf.where(t_p < gamma_t,
    #                     t_p,
    #                     tf.zeros_like(t_p))
    # unsup_loss_weights = tf.cast(tf.shape(S_u_pred)[0]*640*640, dtype=tf.float32) / tf.reduce_sum(labels_u)
    gamma_t = 0.02 + (0.2 - 0.02)*alpha_t
    mask = tf.where(t_p > gamma_t, tf.ones_like(t_p), tf.zeros_like(t_p))
    # t_p, t_t, t_bhat = tf.constant(t_p), tf.constant(t_t), tf.constant(t_bhat)
    # p_loss, _ = balanced_crossentropy_loss(labels_u, s_p)
    p_loss = l1_loss(t_p, s_p, mask)
    # t_loss = l1_loss_wo_mask(t_t, s_t)
    # bhat_loss = l1_loss_wo_mask(t_bhat, s_bhat)
    return p_loss #* unsup_loss_weights

KEYS = ['total_loss', 'pmap_loss', 'tmap_loss', 'bhmap_loss', 'unsup_loss']
def semi_loss(ytrue, ypred, S_u_pred, T_u_pred, alpha_t):
    # unsup_weight=2.0
    loss_dict = db_loss(ytrue, ypred)
    unsup_loss = compute_unsupervise_loss(S_u_pred, T_u_pred, alpha_t)
    loss_dict['unsup_loss'] = unsup_loss
    loss_dict['total_loss'] += unsup_loss
    return loss_dict, loss_dict['total_loss']

if __name__ == '__main__':
    import numpy as np
    ytrue = np.random.random((1, 640, 1824, 3)).astype('float32')
#     p = np.random.random((1, 640, 1824, 1))
    p = np.zeros((1, 640, 1824, 1))
#     t = np.random.random((1, 640, 1824, 1))
    t = np.zeros((1, 640, 1824, 1))
#     bhat = np.random.random((1, 640, 1824, 1))
    bhat = np.ones((1, 640, 1824, 1))
    ypred = np.concatenate([p, t, bhat], -1).astype('float32')
#     print(ypred.shape)
    x = db_loss(ytrue, ypred)
    print(x)
