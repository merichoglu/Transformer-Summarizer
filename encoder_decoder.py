from typing import Tuple
import tensorflow as tf
from transformer_utils import positional_encoding

def FullyConnected(embedding_dim: int, fully_connected_dim: int) -> tf.keras.Model:
    """
    sequential model with two dense layers:
      - dense(fully_connected_dim, relu)
      - dense(embedding_dim)
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(fully_connected_dim, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ]
    )


class EncoderLayer(tf.keras.layers.Layer):
    """
    One encoder layer = multi-head self-attention + ffn
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super(EncoderLayer, self).__init__()
        
        # Support masking in the layer
        self.supports_masking = True

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout_rate
        )

        self.ffn = FullyConnected(embedding_dim, fully_connected_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        """
        x shape: (batch_size, input_seq_len, embedding_dim)
        mask shape: (batch_size, input_seq_len) for padding mask
        """
        self_mha_output = self.mha(
            query=x, value=x, key=x, key_mask=mask, training=training
        )

        skip_x_attention = self.layernorm1(x + self_mha_output)
        ffn_output = self.ffn(skip_x_attention)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        encoder_layer_out = self.layernorm2(skip_x_attention + ffn_output)
        return encoder_layer_out


class Encoder(tf.keras.layers.Layer):
    """
    The entire Encoder =
      - Embedding + positional_encoding
      - stacked encoder layers
    """

    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        input_vocab_size,
        maximum_positional_encoding,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super(Encoder, self).__init__()
        
        # Support masking in the layer
        self.supports_masking = True

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Enable masking in embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_vocab_size, embedding_dim, mask_zero=True
        )
        self.pos_encoding = positional_encoding(
            maximum_positional_encoding, embedding_dim
        )

        self.enc_layers = [
            EncoderLayer(
                embedding_dim,
                num_heads,
                fully_connected_dim,
                dropout_rate,
                layernorm_eps,
            )
            for _ in range(self.num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x


class DecoderLayer(tf.keras.layers.Layer):
    """
    One decoder layer =
      - first multi-head self-attn (with lookahead_mask)
      - second multi-head attn (attending to encoder output, with padding_mask)
      - ffn
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super(DecoderLayer, self).__init__()
        
        # Support masking in the layer
        self.supports_masking = True

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout_rate
        )

        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads, dropout=dropout_rate
        )

        self.ffn = FullyConnected(embedding_dim, fully_connected_dim)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)

        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self,
        x: tf.Tensor,
        enc_output: tf.Tensor,
        training: bool,
        look_ahead_mask: tf.Tensor,
        padding_mask: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        x shape: (batch_size, target_seq_len, embedding_dim)
        enc_output shape: (batch_size, input_seq_len, embedding_dim)
        """
        
        mult_attn_out1, attn_weights_block1 = self.mha1(
            query=x,
            value=x,
            key=x,
            attention_mask=look_ahead_mask,
            training=training,
            return_attention_scores=True,
        )
        Q1 = self.layernorm1(mult_attn_out1 + x)

        mult_attn_out2, attn_weights_block2 = self.mha2(
            query=Q1,
            value=enc_output,
            key=enc_output,
            key_mask=padding_mask,
            training=training,
            return_attention_scores=True,
        )

        mult_attn_out2 = self.layernorm2(Q1 + mult_attn_out2)
        ffn_output = self.ffn(mult_attn_out2)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + mult_attn_out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    """
    The entire Decoder =
      - Embedding + positional_encoding
      - stacked decoder layers
    """

    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        target_vocab_size,
        maximum_positional_encoding,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super(Decoder, self).__init__()
        
        # Support masking in the layer
        self.supports_masking = True

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Enable masking in embedding layer
        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size, embedding_dim, mask_zero=True
        )
        self.pos_encoding = positional_encoding(
            maximum_positional_encoding, embedding_dim
        )

        self.dec_layers = [
            DecoderLayer(
                self.embedding_dim,
                num_heads,
                fully_connected_dim,
                dropout_rate,
                layernorm_eps,
            )
            for _ in range(self.num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self,
        x: tf.Tensor,
        enc_output: tf.Tensor,
        training: bool,
        look_ahead_mask: tf.Tensor,
        padding_mask: tf.Tensor,
    ) -> tuple[tf.Tensor, dict]:
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x=x,
                enc_output=enc_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask,
            )
            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2

        return x, attention_weights