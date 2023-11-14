from collections import defaultdict
from typing import Dict, Set

from typing import Set
import re




import numpy as np


class NaiveBayesClassifier:

    def __init__(self, data, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens = self.tokenize(data)

        self.alt_atheism: Dict[str, int] = defaultdict(int)
        self.soc_religion_christian: Dict[str, int] = defaultdict(int)
        self.comp_graphics: Dict[str, int] = defaultdict(int)
        self.sci_med: Dict[str, int] = defaultdict(int)

        self.extract_words_from_article(0, data, self.alt_atheism)
        self.extract_words_from_article(1, data, self.soc_religion_christian)
        self.extract_words_from_article(2, data, self.comp_graphics)
        self.extract_words_from_article(3, data, self.sci_med)

        self.num_alt_atheism, self.num_soc_religion_christian, self.num_comp_graphics, self.num_sci_med = self.get_labels(
            data)

    def get_labels(self, data):
        list_data = np.array(data.target)
        return len(list_data == 0), len(list_data == 1), len(list_data == 2), len(list_data == 3)

    def tokenize(self, text: str) -> Set[str]:
        text = text.lower()  # Convert to lowercase,
        all_words = re.findall("[a-z0-9']+", text)  # extract the words, and
        return set(all_words)  # remove duplicates.

    def extract_words_from_article(self, category, data, category_dic):
        my_article = data.data[data.target == category]
        for article in my_article:
            for word in self.tokenize(article):
                category_dic[word] += 1


