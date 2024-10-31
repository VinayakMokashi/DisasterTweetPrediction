# Disaster-Related Tweet Classification

This project applies Natural Language Processing (NLP) and Machine Learning (ML) techniques to classify disaster-related tweets, aiming to enhance crisis identification. By analyzing tweets, the model determines whether a tweet pertains to a real disaster or not.

## Project Overview:

Disaster-related information on social media can be invaluable for crisis response. This project uses the "Natural Language Processing with Disaster Tweets" dataset to classify tweets based on whether they describe a real disaster. Through TF-IDF and Word2Vec vectorization, and experimenting with SVM and Naive Bayes models, the project achieves an accuracy of 80%. Model robustness is enhanced by k-fold cross-validation.

## Data Description:

The dataset consists of 7,613 tweets with the following attributes:

__id__: Unique identifier for each tweet

__text__: Content of the tweet

__location__: The tweet's originating location (may be blank)

__keyword__: A keyword related to the tweet (may be blank)

__target__: Indicates if the tweet is disaster-related (1) or not (0)

## Methodology:

### Data Preprocessing:

__Cleaning__: Removed punctuation, special characters, URLs, hashtags, and extra spaces.

__Text Normalization__: Corrected typos, expanded abbreviations, removed stop words, and applied stemming and lemmatization for consistency.

### Vectorization:

__TF-IDF__: Calculates term importance by frequency.

__Word2Vec__: Embeds semantic context of words, capturing word relationships.

### Classification Models:

__Support Vector Machine (SVM) and Naive Bayes__: Used for tweet classification, achieving a balanced accuracy of 80% with TF-IDF. Improvements were made through bagging and boosting techniques.

__Cross-Validation__: Applied k-fold cross-validation (k=5, 10) to ensure model reliability.

## Key Findings:

Achieved 80% accuracy in classifying disaster-related tweets.

Key model improvements were observed with bagging and boosting, highlighting SVM's and Naive Baiye's effectiveness in text classification. Cross-validation proved essential in enhancing model stability. Also Tf-Idf performed better as a vectorization technique.

## Usage:

The repository contains:

__Data Files__:

train.csv: Training dataset.

test.csv: Testing dataset.

__Script__:

Code.ipynb


## Conclusion and Recommendations:

This project demonstrates that combining TF-IDF and Word2Vec with SVM and Naive Bayes classifiers can effectively identify disaster-related tweets. The accuracy achieved suggests these models can support real-time crisis identification.

## Future Scope:

Further improvements could include:

__Deep Learning Models__: Incorporating LSTM or BERT for higher accuracy in text classification.

__Real-Time Prediction__: Adapting the model for live social media data streams.

__Additional Text Features__: Including sentiment analysis for better context understanding.
