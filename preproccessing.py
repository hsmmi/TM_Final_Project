# Tokenization using NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


nltk.download("punkt")


class preproccessing:
    def __init__(self):
        pass

    def preproccess(self, dataset):
        # Tokenization
        tokenized_dataset = self.tokenize(dataset)

        # Normalization
        normalized_dataset = self.normalization(tokenized_dataset)

        # Stemming
        stemmed_dataset = self.stemming(normalized_dataset)

        # Remove stop words
        cleaned_dataset = self.remove_stop_word(stemmed_dataset)

        return cleaned_dataset

    def tokenize(self, dataset: list):
        """
        Input: list of sentences
        Tokenize the dataset using NLTK
        """
        tokenized_dataset = [word_tokenize(sentence) for sentence in dataset]

        return tokenized_dataset

    def normalization(self, dataset):
        """
        Input: list of tokenized sentences
        Normalize the dataset by converting all tokens to lowercase
        """
        normalized_dataset = [
            [token.lower() for token in sentence] for sentence in dataset
        ]
        return normalized_dataset

    def stemming(self, dataset):
        """
        Input: list of normalized sentences
        Stem the dataset using NLTK
        """
        # Initialize Python porter stemmer
        ps = PorterStemmer()

        stemmed_dataset = [
            [ps.stem(token) for token in sentence] for sentence in dataset
        ]

        return stemmed_dataset

    def remove_stop_word(self, dataset):
        """
        Input: list of stemmed sentences
        Remove stop words using NLTK
        """
        # Initialize Python set of stopwords
        stop_words = set(nltk.corpus.stopwords.words("english"))

        cleaned_dataset = [
            [token for token in sentence if token not in stop_words]
            for sentence in dataset
        ]

        return cleaned_dataset
