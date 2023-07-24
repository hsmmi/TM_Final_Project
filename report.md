# Final Project of Text Mining

## 1. Introduction
Semantic Textual Similarity (STS) plays a crucial role in various natural language processing applications, such as question answering, document summarization, and information retrieval. The project aims to develop an STS model using a dataset containing English sentence pairs annotated with human-assigned similarity scores.

## 2. Dataset
The dataset consists of 4500 English sentence pairs for training and 4927 sentence pairs for testing in a "txt" file. Each sentence pair is annotated with a similarity score ranging from 0 (dissimilar) to 5 (completely similar). 

### 2.1 Read the dataset
I read the txt file using pandas and remove ID column.

### 2.2 Data Preprocessing:
First I tokenize the sentences using nltk word_tokenize. Then I normalized the words with lower case and the stemmer. Finally, I removed the stop words.



