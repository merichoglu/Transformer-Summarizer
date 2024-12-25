import tensorflow as tf
from encoder_decoder import *

# The flow of the transformer model is as follows:
# 1. The input sentence is passed through N encoder layers that generates an output for each word/token in the sentence.
# 2. The decoder is given the target sentence and the encoder output to produce the final output.
# 3. The transformer model outputs the predicted target sentence.


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
        encoder_output = self.encoder(input_sentence, training, enc_padding_mask)
        decoder_output, attention_weights = self.decoder(
            target_sentence, encoder_output, training, look_ahead_mask, dec_padding_mask
        )
        final_output = self.final_layer(decoder_output)

        return final_output, attention_weights
