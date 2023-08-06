import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class STS:
    def __init__(self):
        pass

    def calculate_similarity_score(self, sentence1: list, sentence2: list):
        """
        Input: sentence1 (list of floats) [768] (sentence embedding)
                sentence2 (list of floats) [768] (sentence embedding)

        Output: similarity_score (float) [0, 5]
        """
        # Get embeddings for the input sentences
        embeddings1 = np.array(sentence1).reshape(1, -1)
        embeddings2 = np.array(sentence2).reshape(1, -1)

        # Calculate cosine similarity score
        similarity_score = cosine_similarity(embeddings1, embeddings2)[0][0]
        return similarity_score * 5

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Input: X (numpy array) [sentence1, sentence2]
                y (numpy array) [score]

        Method to fine-tune the model using the train data and predict
        the similarity scores for the test data
        """
        # Initialize lists to store predicted and human similarity scores
        predicted_scores = []
        human_scores = []

        sentence1_embeddings = X[0]
        sentence2_embeddings = X[1]
        for i in range(sentence1_embeddings.shape[0]):
            # Get sentences
            s1 = sentence1_embeddings[i]
            s2 = sentence2_embeddings[i]

            # Calculate predicted similarity score
            predicted_score = self.calculate_similarity_score(s1, s2)

            # Append to lists
            predicted_scores.append(predicted_score)
            human_scores.append(y[i])

        # Convert lists to numpy arrays
        predicted_scores = np.array(predicted_scores)
        human_scores = np.array(human_scores)

        # Calculate Pearson correlation
        pearson_correlation = self.evaluate(predicted_scores, human_scores)

        return pearson_correlation

    def predict(self, X: np.ndarray):
        return np.array(
            [self.calculate_similarity_score(x[0], x[1]) for x in X]
        )

    def evaluate(self, predicted_scores: np.ndarray, human_scores: np.ndarray):
        # Calculate Pearson correlation
        pearson_correlation = pd.Series(predicted_scores).corr(
            pd.Series(human_scores), method="pearson"
        )

        return pearson_correlation
