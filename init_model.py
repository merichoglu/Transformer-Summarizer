import tensorflow as tf
from transformer import *
from data_preprocess import *
from transformer_utils import *
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# define model parameters
num_layers = 4  # Increased depth
embedding_dim = 256  # Increased for large vocabulary
fully_connected_dim = 1024  # Increased for better capacity
num_heads = 8  # key_dim = 32 (256/8)
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

# Custom learning rate schedule
learning_rate = CustomSchedule(embedding_dim, warmup_steps=4000)  # Proper warmup
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)

def masked_loss(real, pred, label_smoothing=0.1):
    """
    Masked loss with manual label smoothing implementation
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    # Manual label smoothing
    vocab_size = tf.shape(pred)[-1]
    confidence = 1.0 - label_smoothing
    smooth_positives = confidence
    smooth_negatives = label_smoothing / tf.cast(vocab_size - 1, tf.float32)

    # One-hot encoding
    one_hot = tf.one_hot(real, vocab_size)
    # Label smoothing
    smoothed_labels = one_hot * smooth_positives + (1 - one_hot) * smooth_negatives

    # Compute cross-entropy loss
    loss_ = tf.nn.softmax_cross_entropy_with_logits(
        labels=smoothed_labels, logits=pred
    )

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

    with tf.GradientTape() as tape:
        # Create masks
        enc_padding_mask = create_padding_mask(inp)
        combined_mask = create_combined_mask(tar_inp)
        dec_padding_mask = create_padding_mask(inp)

        # Pass all arguments correctly to the model
        predictions, _ = model(
            input_sentence=inp,
            target_sentence=tar_inp,
            training=True,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask,
        )
        
        # Compute the loss
        loss = masked_loss(tar_real, predictions)

    # Compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Apply gradient clipping
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
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
    combined_mask = create_combined_mask(output)
    dec_padding_mask = create_padding_mask(encoder_input)

    # Run the prediction of the next word with the transformer model
    predictions, attention_weights = model(
        input_sentence=encoder_input,
        target_sentence=output,
        training=False,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=combined_mask,
        dec_padding_mask=dec_padding_mask,
    )

    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    return predicted_id

def summarize(model, input_document: tf.Tensor, temperature: float = 0.8, top_k: int = 40) -> str:
    """
    A function for summarization using the transformer model with improved decoding
    Arguments:
        input_document (tf.Tensor): Input data to summarize
        temperature (float): Sampling temperature for diversity
        top_k (int): Top-k sampling for better quality
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
        # Get predictions for next word
        enc_padding_mask = create_padding_mask(encoder_input)
        combined_mask = create_combined_mask(output)
        dec_padding_mask = create_padding_mask(encoder_input)

        predictions, _ = model(
            input_sentence=encoder_input,
            target_sentence=output,
            training=False,
            enc_padding_mask=enc_padding_mask,
            look_ahead_mask=combined_mask,
            dec_padding_mask=dec_padding_mask,
        )

        predictions = predictions[:, -1:, :]

        # Temperature scaling and top-k sampling
        predictions = predictions / temperature

        # Get top-k tokens
        top_k_predictions, top_k_indices = tf.nn.top_k(predictions, k=min(top_k, tf.shape(predictions)[-1]))
        top_k_probs = tf.nn.softmax(top_k_predictions, axis=-1)

        # Sample from top-k distribution
        sampled_index = tf.random.categorical(tf.math.log(top_k_probs[0]), 1)
        predicted_id = tf.gather(top_k_indices[0], sampled_index, axis=1)
        predicted_id = tf.cast(predicted_id, tf.int32)
        predicted_id = tf.reshape(predicted_id, [1, 1])  # Ensure correct shape

        output = tf.concat([output, predicted_id], axis=-1)

        if predicted_id == tokenizer.word_index["[EOS]"]:
            break

    return tokenizer.sequences_to_texts(output.numpy())[0]

## Take an example from the validation set for monitoring
test_example = 0
val_summary = summary[train_size + test_example]  # Use validation data
val_document = document[train_size + test_example]  # Use validation data

# Define the number of epochs
epochs = 10

# Add early stopping
best_loss = float('inf')
patience = 3
patience_counter = 0

print("Starting training...")

# Training loop
for epoch in tqdm(range(epochs), desc="Training Progress"):
    start = time.time()
    train_loss.reset_state()

    # Create progress bar for batches
    batch_progress = tqdm(enumerate(dataset), desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    for batch, (inp, tar) in batch_progress:
        train_step(transformer, inp, tar)
        batch_progress.set_postfix({"Loss": f"{train_loss.result():.4f}"})

    epoch_loss = train_loss.result()
    print(f"\nEpoch {epoch+1}/{epochs}, Loss {epoch_loss:.4f}")
    losses.append(epoch_loss)

    print(f"Time taken for one epoch: {time.time() - start:.2f} sec")
    
    # Early stopping check
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
        # Save best model
        transformer.save_weights("transformer_best_weights.weights.h5")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    
    print("Example summarization on validation set:")
    print("  True summarization:")
    print(f"    {val_summary}")
    print("  Predicted summarization:")
    try:
        pred_summary = summarize(transformer, val_document)
        print(f"    {pred_summary}")
    except Exception as e:
        print(f"    Error in prediction: {e}")
    print()

    # Validation loss evaluation
    if epoch % 2 == 0:  # Every 2 epochs
        val_loss = tf.keras.metrics.Mean()
        for val_inp, val_tar in val_dataset:
            val_tar_inp = val_tar[:, :-1]
            val_tar_real = val_tar[:, 1:]

            enc_padding_mask = create_padding_mask(val_inp)
            combined_mask = create_combined_mask(val_tar_inp)
            dec_padding_mask = create_padding_mask(val_inp)

            predictions, _ = transformer(
                input_sentence=val_inp,
                target_sentence=val_tar_inp,
                training=False,
                enc_padding_mask=enc_padding_mask,
                look_ahead_mask=combined_mask,
                dec_padding_mask=dec_padding_mask,
            )

            val_loss_value = masked_loss(val_tar_real, predictions)
            val_loss(val_loss_value)

        print(f"Validation Loss: {val_loss.result():.4f}")
        print()

# Plot the loss
plt.figure(figsize=(10, 6))
plt.plot(losses, 'b-', linewidth=2)
plt.title('Training Loss Over Time')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, alpha=0.3)
plt.show()

# Save the final model
transformer.save_weights("transformer_final_weights.weights.h5")
print("Training completed and model saved!")