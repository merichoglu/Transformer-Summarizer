import pandas as pd
import data_utils
import tensorflow as tf

# load the data
data_dir = "data/corpus"

train_data, test_data = data_utils.get_train_test_data(data_dir=data_dir)

# Take one example from the dataset and print it
example_summary, example_dialogue = train_data.iloc[10]
print(f"Dialogue:\n{example_dialogue}")
print(f"\nSummary:\n{example_summary}")

# Preprocess the data
document, summary = data_utils.preprocess(train_data)
document_test, summary_test = data_utils.preprocess(test_data)

filters = '!"#$%&()*+,-./:;<=>?@\\^_`{|}~\t\n'
oov_token = "[UNK]"

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    filters=filters, oov_token=oov_token, lower=False
)
documents_and_summaries = pd.concat([document, summary], ignore_index=True)
tokenizer.fit_on_texts(documents_and_summaries)

inputs = tokenizer.texts_to_sequences(document)
targets = tokenizer.texts_to_sequences(summary)
vocab_size = len(tokenizer.word_index) + 1
# print(f"Vocab size: {vocab_size}") # 34250

# now pad the tokenized sequences to make them the same length
encoder_maxlen = 150
decoder_maxlen = 50

inputs = tf.keras.preprocessing.sequence.pad_sequences(
    inputs, maxlen=encoder_maxlen, padding="post", truncating="post"
)

targets = tf.keras.preprocessing.sequence.pad_sequences(
    targets, maxlen=decoder_maxlen, padding="post", truncating="post"
)

inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)

# Create the dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64

dataset = (
    tf.data.Dataset.from_tensor_slices((inputs, targets))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
