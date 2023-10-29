"""
Script with data transforms.
"""
import re
import string

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm


def apply_transforms(dataset):
    """
    Apply transforms to the dataset.
    """

    print('1. Cleaning text...')
    dataset.toxic_X = [clean_text(text) for text in tqdm(dataset.toxic_X, desc='Cleaning toxic text')]
    dataset.normal_X = [clean_text(text) for text in tqdm(dataset.normal_X, desc='Cleaning normal text')]

    print('2. Tokenizing text...')
    dataset.toxic_X = [word_tokenize(text) for text in tqdm(dataset.toxic_X, desc='Tokenizing toxic text')]
    dataset.normal_X = [word_tokenize(text) for text in tqdm(dataset.normal_X, desc='Tokenizing normal text')]

    print('3. Removing stopwords...')
    stop_words = set(stopwords.words('english'))
    dataset.toxic_X = [[word for word in text if word not in stop_words] for text in
                       tqdm(dataset.toxic_X, desc='Removing stopwords from toxic text')]
    dataset.normal_X = [[word for word in text if word not in stop_words] for text in
                        tqdm(dataset.normal_X, desc='Removing stopwords from normal text')]

    print('4. Lemmatizing text...')
    lemmatizer = WordNetLemmatizer()
    dataset.toxic_X = [[lemmatizer.lemmatize(word) for word in text] for text in
                       tqdm(dataset.toxic_X, desc='Lemmatizing toxic text')]
    dataset.normal_X = [[lemmatizer.lemmatize(word) for word in text] for text in
                        tqdm(dataset.normal_X, desc='Lemmatizing normal text')]

    print('5. Removing empty texts...')
    dataset.toxic_X, dataset.toxic_y = remove_empty_texts(zip(dataset.toxic_X, dataset.toxic_y),
                                                          desc='Removing empty texts from toxic text')
    dataset.normal_X, dataset.normal_y = remove_empty_texts(zip(dataset.normal_X, dataset.normal_y),
                                                            desc='Removing empty texts from normal text')

    print("Transfomed data sample:")
    print(
        pd.DataFrame({'text': dataset.toxic_X + dataset.normal_X, 'label': dataset.toxic_y + dataset.normal_y})
        .sample(n=5))

    print('All done.')


def clean_text(text):
    """
    Clean the text.
    """
    # Remove non-ascii characters
    text = text.encode("ascii", errors="ignore").decode()
    # Remove urls
    text = re.sub(r'http\S+', '', text)
    # Remove html tags
    text = re.sub(r'<.*?>', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Lowercase
    text = text.lower()
    return text


def remove_empty_texts(data, desc):
    """
    Remove empty texts.
    """
    X, y = [], []
    for text, label in tqdm(data, desc=desc):
        if len(text) > 0:
            X.append(text)
            y.append(label)
    return X, y
