# Final Project of Text Mining

## 1. Introduction
Semantic Textual Similarity (STS) plays a crucial role in various natural language processing applications, such as question answering, document summarization, and information retrieval. The project aims to develop an STS model using a dataset containing English sentence pairs annotated with human-assigned similarity scores.

## 2. Dataset
The dataset consists of 4500 English sentence pairs for training and 4927 sentence pairs for testing in a "txt" file. Each sentence pair is annotated with a similarity score ranging from 0 (dissimilar) to 5 (completely similar). 

## 3. Data Preprocessing

In this part, we focused on preparing the dataset for the Semantic Textual Similarity (STS) model by performing data preprocessing. The dataset consists of pairs of sentences (s1 and s2) and their associated semantic similarity scores. The following steps were undertaken for data preprocessing:

1. Convert Text to Lowercase:
To ensure uniformity and avoid considering the case of words as a factor in semantic similarity, we converted the entire text to lowercase. This process makes the comparison between sentences case-insensitive.

2. Remove Special Characters and Digits:
Special characters and digits often add noise and irrelevant information to the text. Therefore, we utilized regular expressions to remove these elements, allowing the model to focus on the meaningful content of the sentences.

3. Tokenization:
Tokenization is a critical step in natural language processing, as it breaks the text into individual words (tokens). We employed the Natural Language Toolkit (NLTK) library to tokenize the sentences, enabling better understanding and processing of the textual data.

4. Remove Stopwords:
Stopwords are common words in a language that do not carry significant meaning in the context of the text. Examples of stopwords include "the," "is," "a," and "in." By removing these words, we reduce the dimensionality of the data and concentrate on the more informative content.

The preprocessed datasets for training and testing were then saved for future use in subsequent parts of the project.

Sentence embedding: In natural language processing, a sentence embedding refers to a numeric representation of a sentence in the form of a vectors of real numbers which encodes meaningful semantic information.