from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Remove stopwords and punctuation, and convert to lowercase
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    # Perform stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return sentences, words

def extract_features(sentences):
    # Convert sentences to a TF-IDF matrix of features
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(sentences)

    return features

def summarize_text(text, num_sentences=3):
    sentences, words = preprocess_text(text)
    features = extract_features(sentences)

    # Compute sentence scores based on TF-IDF values
    sentence_scores = features.sum(axis=1)

    # Sort the sentences by score in descending order
    ranked_sentences = sorted(((score, index) for index, score in enumerate(sentence_scores)), reverse=True)

    # Get the top N sentences for the summary
    top_sentences = sorted([index for score, index in ranked_sentences[:num_sentences]])

    # Build the summary
    summary = ' '.join([sentences[i] for i in top_sentences])

    return summary

# Example usage
text = """
Text summarization is the process of distilling the most important information from a source text into a shorter version, without losing the key concepts and main ideas. It can be a useful technique for dealing with large amounts of text, such as news articles, research papers, or online documents.

There are various approaches to text summarization, including extractive and abstractive methods. Extractive summarization involves selecting and combining existing sentences or phrases from the source text to create a summary. Abstractive summarization, on the other hand, involves generating new sentences that capture the meaning of the original text.

In this example, we'll focus on extractive text summarization using feature extraction. We'll use the TF-IDF (Term Frequency-Inverse Document Frequency) approach to represent the sentences as numerical features and compute their importance scores. Then, we'll select the top-scoring sentences to form the summary.

Let's see how it works!
"""

summary = summarize_text(text)
print(summary)
