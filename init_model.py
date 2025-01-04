from transformer import *
from data_preprocess import *
from transformer_utils import *
import matplotlib.pyplot as plt
import time

# define model parameters
num_layers = 2
embedding_dim = 128
fully_connected_dim = 128
num_heads = 2
positional_encoding_length = 256

# Initialize the model
transformer = Transformer(
    num_layers,
    embedding_dim,
    num_heads,
    fully_connected_dim,
    vocab_size,
    vocab_size,
    positional_encoding_length,
    positional_encoding_length,
)


# Initialize the optimizer. THIS PIECE IS FROM THE ORIGINAL TRANSFORMER PAPER
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(embedding_dim)

optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)


def masked_loss(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name="train_loss")
losses = []  # store the loss for plotting


@tf.function
def train_step(model, inp: tf.Tensor, tar: tf.Tensor) -> None:
    """
    One training step for the transformer
    Arguments:
        inp (tf.Tensor): Input data to summarize (encoder input)
        tar (tf.Tensor): Target (summary, decoder input)
    Returns:
        None
    """
    tar_inp = tar[:, :-1]  # Remove the last token (target input)
    tar_real = tar[:, 1:]  # Remove the first token (target output)

    # Create masks
    enc_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar_inp)[1])
    dec_padding_mask = create_padding_mask(inp)

    with tf.GradientTape() as tape:
        # Pass all arguments correctly to the model
        predictions, _ = model(
            input_sentence=inp,
            target_sentence=tar_inp,
            training=True,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=look_ahead_mask,
            dec_padding_mask=dec_padding_mask,
        )
        # Compute the loss
        loss = masked_loss(tar_real, predictions)

    # Compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Update training loss
    train_loss(loss)


# next word prediction helper function
def next_word(model, encoder_input: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    """
    Helper function for summarization that uses the model to predict just the next word.
    Arguments:
        encoder_input (tf.Tensor): Input data to summarize
        output (tf.Tensor): (incomplete) target (summary)
    Returns:
        predicted_id (tf.Tensor): The id of the predicted word
    """

    enc_padding_mask = create_padding_mask(encoder_input)
    look_ahead_mask = create_look_ahead_mask(tf.shape(output)[1])
    dec_padding_mask = create_padding_mask(encoder_input)

    # Run the prediction of the next word with the transformer model
    predictions, attention_weights = model(
        input_sentence=encoder_input,
        target_sentence=output,
        training=False,  # pass "False" as a keyword
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=look_ahead_mask,
        dec_padding_mask=dec_padding_mask,
    )

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    return predicted_id


def summarize(model, input_document: tf.Tensor) -> str:
    """
    A function for summarization using the transformer model
    Arguments:
        input_document (tf.Tensor): Input data to summarize
    Returns:
        _ (str): The summary of the input_document
    """
    input_document = tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(
        input_document, maxlen=encoder_maxlen, padding="post", truncating="post"
    )
    encoder_input = tf.expand_dims(input_document[0], 0)

    output = tf.expand_dims([tokenizer.word_index["[SOS]"]], 0)

    for i in range(decoder_maxlen):
        predicted_id = next_word(model, encoder_input, output)
        output = tf.concat([output, predicted_id], axis=-1)

        if predicted_id == tokenizer.word_index["[EOS]"]:
            break

    return tokenizer.sequences_to_texts(output.numpy())[
        0
    ]  # since there is just one translated document


## train the model and plot the loss
## Take an example from the test set, to monitor it during training
test_example = 0
true_summary = summary_test[test_example]
true_document = document_test[test_example]

# Define the number of epochs
epochs = 20

# Training loop
for epoch in range(epochs):

    start = time.time()
    train_loss.reset_state()
    number_of_batches = len(list(enumerate(dataset)))

    for batch, (inp, tar) in enumerate(dataset):
        print(f"Epoch {epoch+1}, Batch {batch+1}/{number_of_batches}", end="\r")
        train_step(transformer, inp, tar)

    print(f"Epoch {epoch+1}, Loss {train_loss.result():.4f}")
    losses.append(train_loss.result())

    print(f"Time taken for one epoch: {time.time() - start} sec")
    print("Example summarization on the test set:")
    print("  True summarization:")
    print(f"    {true_summary}")
    print("  Predicted summarization:")
    print(f"    {summarize(transformer, true_document)}\n")

plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.show()

# Save the model
transformer.save_weights("transformer_weights")
