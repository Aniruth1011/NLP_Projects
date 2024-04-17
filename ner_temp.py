import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# Sample data for demonstration
sentences = [
    "Apple is expected to release a new iPhone next month.",
    "The CEO of Google, Sundar Pichai, announced a major investment.",
    "Tesla's stock price surged after Elon Musk's tweet.",
]

# Tokenize the sentences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
max_len = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

# Create NER model
vocab_size = len(word_index) + 1
embedding_dim = 50
ner_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=True)),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(50, activation='relu')),
    tf.keras.layers.Dense(len(tfa.text.tagging.en.TNT_POS_TAGGER_TOKENS), activation='softmax')
])

# Compile NER model
ner_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train NER model (dummy training for demonstration)
ner_model.fit(padded_sequences, np.zeros((len(padded_sequences), max_len, len(tfa.text.tagging.en.TNT_POS_TAGGER_TOKENS))), epochs=10, batch_size=32)

# Predict NER tags for each sentence
ner_tags = ner_model.predict(padded_sequences)

# Decode predicted NER tags
decoded_tags = tfa.text.tagging.en.TNT_POS_TAGGER_DECODE_MAP[np.argmax(ner_tags, axis=-1)]

# Identify critical words aiding in NER identification
critical_words = {}
for i, sentence in enumerate(sentences):
    for j, word in enumerate(sentence.split()):
        if decoded_tags[i][j] != 'O':
            if decoded_tags[i][j] not in critical_words:
                critical_words[decoded_tags[i][j]] = []
            critical_words[decoded_tags[i][j]].append(word)

# Print critical words aiding in NER identification
for ner_type, words in critical_words.items():
    print(f"{ner_type}: {words}")
