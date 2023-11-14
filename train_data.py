from sklearn.datasets import fetch_20newsgroups

from typing import Set
import re

from NaiveBayesClassifier import NaiveBayesClassifier


def tokenize(text: str) -> Set[str]:
    text = text.lower()  # Convert to lowercase,
    all_words = re.findall("[a-z0-9']+", text)  # extract the words, and
    return set(all_words)  # remove duplicates.


assert tokenize("Data Science is science") == {"data", "science", "is"}

# Load the 20newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
train_data = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42)
# print(type(train_data), type(test_data))
print(type(test_data.data[0]))
print(train_data.target)


a = NaiveBayesClassifier(train_data)