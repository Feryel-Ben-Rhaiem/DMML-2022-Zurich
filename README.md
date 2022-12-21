# Data Mining and Machine Learning 2022
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Logo_Université_de_Lausanne.svg/1280px-Logo_Université_de_Lausanne.svg.png" width="250" height="100" /> <br>

**Group** : Zurich <br>
**Group members** : Emiliya AKHMETOVA, Feryel BEN RHAIEM

#### This github repo contains the following folders:
* **code** : all codes related to the project (models, functions etc) 
* **data** : test, train and validation data 
* **documents** : littereture review for this project 

## 1. Project Description : 
As a part of Data Mining and Machine Learning course at Master's programm in University of Lausanne we were asked to find the best model to predict the difficulty level of french texts. The purpose of this project is therefore to help 

You have noticed that to improve one’s skills in a new foreign language, it is important to read texts in that language. These texts have to be at the reader’s language level. However, it is difficult to find texts that are close to someone’s knowledge level (A1 to C2). You have decided to build a model for English speakers that predicts the difficulty of a French written text. This can be then used, e.g., in a recommendation system, to recommend texts, e.g, recent news articles that are appropriate for someone’s language level. If someone is at A1 French level, it is inappropriate to present a text at B2 level, as she won’t be able to understand it. Ideally, a text should have many known words and may have a few words that are unknown so that the person can improve.

<br>
There 6 different levels: 
* A1 : begginer 
* A2 : elementary
* B1 : intermediate
* B2 : upper intermediate 
* C1 : advanced
* C2 : native

## 2. Methodology 
In this project we have tried several algorithms. <br>
First, we tried several **text cleaning and pre-processing** algorithms like : **tokenization** and **stop-words removal**. <br>
We have also tried some text classification techniques like: **k-nearest neighbor**, **decision tree** and **random forest**. <br>
Finaly, we decided to apply **BERT multilingual model**, which gave us the best results in accuracy of our predictions. <br> 

### Tokenization 
[Tokenization](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#tokenization) is the process of breaking down a stream of text into words, phrases, symbols, or any other meaningful elements called tokens. The main goal of this step is to extract individual words in a sentence. 

### Stop-words
A [stop word](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/) is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query. 

### K-nearest neighbor 
In machine learning, the [k-nearest neighbors algorithm (kNN)](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#k-nearest-neighbor) is a non-parametric technique used for classification. This method is used in Natural-language processing (NLP) as a text classification technique in many researches in the past decades.

### Decision Tree
One of earlier classification algorithm for text and data mining is [decision tree](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#decision-tree). Decision tree classifiers (DTC's) are used successfully in many diverse areas of classification. The structure of this technique includes a hierarchical decomposition of the data space (only train dataset). Decision tree as classification task was introduced by D. Morgan and developed by JR. Quinlan. The main idea is creating trees based on the attributes of the data points, but the challenge is determining which attribute should be in parent level and which one should be in child level. To solve this problem, De Mantaras introduced statistical modeling for feature selection in tree.

### Random Forests
[Random forests](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#random-forest) or random decision forests technique is an ensemble learning method for text classification. This method was introduced by T. Kam Ho in 1995 for first time which used t trees in parallel. This technique was later developed by L. Breiman in 1999 that they found converged for RF as a margin measure.

### BERT multilingual
[BERT](https://huggingface.co/bert-base-multilingual-cased) is a transformers model pretrained on a large corpus of multilingual data in a self-supervised fashion.

## 3. Results 

### Final results <br>
Accuracy = 0.55583

### First attempt 
Firstly we simply built 4 models with the use of [TfidfVectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html):
* Logistic Regression
* kNN
* Decision Tree
* Random Forests <br>

Here are the results we obtained: 

|           | Logistic Regression | kNN   | Decision Tree | Random Forests |
| ----------| ------------------- | ----- | ------------- | -------------  |
| Precision |         0.45        | 0.38  |      0.31     |      0.40      |
| Recall    |         0.45        | 0.31  |      0.32     |      0.40      |
| F1 score  |         0.45        | 0.29  |      0.30     |      0.39      |
| Accuracy  |         0.45        | 0.31  |      0.32     |      0.40      |
