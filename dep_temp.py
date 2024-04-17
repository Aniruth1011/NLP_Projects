import tensorflow as tf
from nltk.parse import DependencyGraph
from nltk.tokenize import word_tokenize

# Sample sentences
sentences = [
    "The movie was great and the acting was superb.",
    "I didn't like the plot, but the cinematography was impressive."
]

# Load a pre-trained sentiment classifier (you can replace this with your own model)
sentiment_model = tf.keras.models.load_model('sentiment_model.h5')

# Function to extract sentiment from dependency tree
def analyze_sentiment(sentence):
    tokens = word_tokenize(sentence)
    dep_tree = DependencyGraph(sentence)
    sentiment_score = 0
    for token in dep_tree.nodes.values():
        if token['word'] and token['word'] in tokens:
            # Assuming sentiment scores are stored in the model's output
            sentiment_score += sentiment_model.predict(tf.expand_dims(token['word'], axis=0))[0]
    return sentiment_score

# Analyze sentiment for each sentence
for sentence in sentences:
    sentiment = analyze_sentiment(sentence)
    print(f"Sentence: {sentence}\nSentiment Score: {sentiment}\n")
