import tensorflow as tf

def prepro_for_softmax(logits, mask):
    ''' Make the indexes of the mask values of 1 and indexes of non mask 0
        Set huge neg number(-1e9) in padding area
    '''
    assert tensor.get_shape().ndims == mask.get_shape().ndims
    # filter out the padding area as 1, the index area becomes 0
    new_mask = tf.subtract(tf.constant(1.0), tf.cast(mask, tf.float32))
    paddings_mask = tf.multiply(new_mask, tf.constant(-1e9))
    masked_logits = tf.where(mask, logits, paddings_mask)
    return masked_logits
