import tensorflow as tf
import numpy as np

# Sample data
jumbled_sentences = [
    "sentence jumbled is This",
    "order correct the to need we"
]
correct_sentences = [
    "This is a jumbled sentence",
    "We need to correct the order"
]

# Tokenize sentences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(jumbled_sentences + correct_sentences)
jumbled_seqs = tokenizer.texts_to_sequences(jumbled_sentences)
correct_seqs = tokenizer.texts_to_sequences(correct_sentences)

# Pad sequences
max_len = max(max(len(seq) for seq in jumbled_seqs), max(len(seq) for seq in correct_seqs))
jumbled_seqs = tf.keras.preprocessing.sequence.pad_sequences(jumbled_seqs, maxlen=max_len, padding='post')
correct_seqs = tf.keras.preprocessing.sequence.pad_sequences(correct_seqs, maxlen=max_len, padding='post')

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=max_len),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(tokenizer.word_index) + 1, activation='softmax'))
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(jumbled_seqs, correct_seqs, epochs=10, batch_size=32)

# Function to rearrange jumbled sentence
def rearrange(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    reordered_seq = model.predict(padded_seq)
    reordered_tokens = [tokenizer.index_word[np.argmax(token)] for token in reordered_seq[0]]
    reordered_sentence = ' '.join(token for token in reordered_tokens if token != 'PAD')
    return reordered_sentence

# Test the model
for jumbled_sentence in jumbled_sentences:
    rearranged_sentence = rearrange(jumbled_sentence)
    print(f"Jumbled Sentence: {jumbled_sentence}")
    print(f"Rearranged Sentence: {rearranged_sentence}\n")
