import numpy as np
import tensorflow as tf


def positional_encoding(positions: int, d_model: int) -> tf.Tensor:
    """
    Precomputes a matrix with all the positional encodings

    Arguments:
        positions (int): Maximum number of positions to be encoded
        d_model (int): Encoding size

    Returns:
        pos_encoding (tf.Tensor): A matrix of shape (1, position, d_model) with the positional encodings
    """

    position = np.arange(positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2

    # initialize a matrix angle_rads of all the angles
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = position * angle_rates

    # apply sin to even indices in the array
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(decoder_token_ids) -> tf.Tensor:
    """
    Creates a matrix mask for the padding cells

    Arguments:
        decoder_token_ids (matrix like): matrix of size (n, m)

    Returns:
        mask (tf.Tensor): Tensor with the padding mask of size (n, 1, m)
    """
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
    return seq[:, tf.newaxis, :]


def create_look_ahead_mask(size: int) -> tf.Tensor:
    """
    Create a mask to hide the subsequent words

    Arguments:
        size (int): Size of the mask

    Returns:
        mask (tf.Tensor): Tensor with the look ahead mask
    """
    mask = tf.linalg.band_part(tf.ones((1, size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: tf.Tensor = None
) -> tf.Tensor:
    """
    Scaled dot product attention

    Arguments:
        query (tf.Tensor): Query tensor of shape (..., seq_len_q, depth)
        key (tf.Tensor): Key tensor of shape (..., seq_len_k, depth)
        value (tf.Tensor): Value tensor of shape (..., seq_len_v, depth)
        mask (tf.Tensor): Mask tensor of shape (..., seq_len_q, seq_len_k)

    Returns:
        output (tf.Tensor): Tensor with the scaled dot product attention of shape (..., seq_len_q, depth)
        attention_weights (tf.Tensor): Tensor with the attention weights of shape (..., seq_len_q, seq_len_k)
    """
    # just apply the formula softmax((QK^T)/sqrt(dk))V
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (1.0 - mask) * -1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights
