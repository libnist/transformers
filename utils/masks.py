# Import libraries
import tensorflow as tf
import tensorflow_probability as tfp

# The function below will produce masks for our encoder input
# sequence in order to avoid unkown tokens


def create_padding_mask(seq):
    """
    seq is a sequence of tokens and the output will be the masks to prevent
    the effect of UNK tokens. This mask is to be used in MultiHeadAttention.
    """
    seq = tf.cast(tf.math.not_equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, :]  # (batch_size, 1, seq_len)

# The function below produces masks for our decoder input
# sequence in order to predict based on tokens behind


def create_look_ahead_mask(size):
    """
    This function creates a mask in order to teach a model that only uses the
    past tokens in the decoder of the transformer.
    """
    n = (size * (size+1) // 2)
    mask = tfp.math.fill_triangular(
        tf.ones((n,), dtype=tf.float32), upper=False)
    return mask
