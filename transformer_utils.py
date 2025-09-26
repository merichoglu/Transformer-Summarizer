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


def create_padding_mask(token_ids: tf.Tensor) -> tf.Tensor:
    """
    Creates a padding mask for encoder-decoder attention

    Arguments:
        token_ids (tf.Tensor): tensor of shape (batch, seq_len)

    Returns:
        mask (tf.Tensor): Tensor with shape (batch, seq_len)
    """
    # For tf.keras.layers.MultiHeadAttention, mask should be (batch_size, seq_len)
    mask = tf.cast(tf.equal(token_ids, 0), dtype=tf.bool)
    return mask


def create_look_ahead_mask(size: int) -> tf.Tensor:
    """
    Create a mask to hide subsequent words

    Arguments:
        size (int): target sequence length

    Returns:
        mask (tf.Tensor): Tensor with shape (size, size)
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return tf.cast(mask, dtype=tf.bool)


def create_combined_mask(decoder_input: tf.Tensor) -> tf.Tensor:
    """
    Creates a combined mask for the decoder self-attention:
      - look-ahead mask (to mask future tokens)
      - decoder padding mask (to mask <pad> tokens)

    Arguments:
        decoder_input (tf.Tensor): decoder input of shape (batch, target_seq_len)

    Returns:
        combined_mask (tf.Tensor): Tensor with shape (batch, target_seq_len, target_seq_len)
    """
    seq_len = tf.shape(decoder_input)[1]
    look_ahead = create_look_ahead_mask(seq_len)
    dec_padding = create_padding_mask(decoder_input)

    # Expand dec_padding to match look_ahead shape: (batch, seq_len, seq_len)
    dec_padding = dec_padding[:, tf.newaxis, :] | dec_padding[:, :, tf.newaxis]

    # Use logical OR for combining boolean masks
    return tf.logical_or(look_ahead, dec_padding)
    

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
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        # Convert boolean mask to float and apply
        scaled_attention_logits += tf.cast(mask, dtype=tf.float32) * -1e9

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights