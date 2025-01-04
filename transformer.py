import tensorflow as tf
from encoder_decoder import *

class Transformer(tf.keras.Model):
    """
    Complete transformer with an encoder and a decoder
    """

    def __init__(
        self,
        num_layers,
        embedding_dim,
        num_heads,
        fully_connected_dim,
        input_vocab_size,
        target_vocab_size,
        max_positional_encoding_input,
        max_positional_encoding_target,
        dropout_rate=0.1,
        layernorm_eps=1e-6,
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers,
            embedding_dim,
            num_heads,
            fully_connected_dim,
            input_vocab_size,
            max_positional_encoding_input,
            dropout_rate,
            layernorm_eps,
        )
        self.decoder = Decoder(
            num_layers,
            embedding_dim,
            num_heads,
            fully_connected_dim,
            target_vocab_size,
            max_positional_encoding_target,
            dropout_rate,
            layernorm_eps,
        )

        self.final_layer = tf.keras.layers.Dense(
            target_vocab_size, activation="softmax"
        )

    def call(
        self,
        input_sentence: tf.Tensor,
        target_sentence: tf.Tensor,
        training: bool,
        enc_padding_mask: tf.Tensor,
        look_ahead_mask: tf.Tensor,
        dec_padding_mask: tf.Tensor,
    ) -> tf.Tensor:
        """
        Forward pass for the complete transformer
        """
        # encode
        encoder_output = self.encoder(
            x=input_sentence,  
            training=training, 
            mask=enc_padding_mask
        )

        # decode
        decoder_output, attention_weights = self.decoder(
            x=target_sentence,
            enc_output=encoder_output,
            training=training, 
            look_ahead_mask=look_ahead_mask,
            padding_mask=dec_padding_mask,
        )

        # final
        final_output = self.final_layer(decoder_output)
        return final_output, attention_weights
