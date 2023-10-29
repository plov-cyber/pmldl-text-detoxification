"""
Script with data transforms.
"""
import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def apply_transforms(dataset):
    """
    Apply transforms to the dataset.
    """

    print('1. Cleaning text...')
    dataset.data.toxic = dataset.data.toxic.apply(clean_text)
    dataset.data.normal = dataset.data.normal.apply(clean_text)

    print('2. Tokenizing text...')
    dataset.data.toxic = dataset.data.toxic.apply(word_tokenize)
    dataset.data.normal = dataset.data.normal.apply(word_tokenize)

    print('3. Removing stopwords...')
    stop_words = set(stopwords.words('english'))
    dataset.data.toxic = dataset.data.toxic.apply(lambda text: [word for word in text if word not in stop_words])
    dataset.data.normal = dataset.data.normal.apply(lambda text: [word for word in text if word not in stop_words])

    print('4. Lemmatizing text...')
    lemmatizer = WordNetLemmatizer()
    dataset.data.toxic = dataset.data.toxic.apply(lambda text: [lemmatizer.lemmatize(word) for word in text])
    dataset.data.normal = dataset.data.normal.apply(lambda text: [lemmatizer.lemmatize(word) for word in text])

    print("Transfomed data sample:")
    print(dataset.data.sample(n=5))

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
    # Replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Lowercase
    text = text.lower()

    return text
