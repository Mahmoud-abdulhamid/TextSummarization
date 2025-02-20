# Text Summarization App

This is a simple text summarization application built using Natural Language Processing (NLP) techniques. The app processes a given text, tokenizes it, removes stopwords, performs stemming, and extracts features using TF-IDF (Term Frequency-Inverse Document Frequency) to generate a summary.

## Features

- **Preprocessing**: Tokenizes text into words and sentences, removes stopwords, and applies stemming.
- **TF-IDF Feature Extraction**: Converts the input text into a matrix of TF-IDF features.
- **Text Summarization**: Ranks sentences based on their TF-IDF scores and selects the top ones to generate a summary.

## Technologies Used

- **NLTK**: For text preprocessing, tokenization, stopwords removal, and stemming.
- **Scikit-learn**: For feature extraction using the TF-IDF vectorizer.

## Installation

To run this app, you'll need to install the necessary Python libraries:

```bash
pip install nltk scikit-learn
```

## Output
This will print a summary of the provided text.

## Developer
This application was developed by Mahmoud Abdelhamid.

## License
This project is licensed under the MIT License.

## Copyright
© 2025 Mahmoud Abdelhamid. All rights reserved.
