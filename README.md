# DMML 2022 - Detecting the difficulty level of French text
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Logo_Université_de_Lausanne.svg/1280px-Logo_Université_de_Lausanne.svg.png" width="250" height="100"/> <br>
 </p>

**Group** : Zurich <br>
**Group members** : Emiliya AKHMETOVA, Feryel BEN RHAIEM

#### This github repository contains the following folders:
* **Code** : contains 6 .py files for Exploratory Data Analysis (EDA), data preparation, 4 models and their functions and final BERT model and its functions. 
* **Data** : contains training data, unlabelled test data and enriched training data (new texts were added to existing train data to enlarge our dataset to obtain better results) 
* **Final Submission** : contains one .csv file with ids of our french texts and predicted language difficulty level for final submission on keggle 

## 1. Project Description  
As a part of Data Mining and Machine Learning course at Master's programm in University of Lausanne we were asked to construct the best model to predict the difficulty level of french texts. The utility of such model could be used in a recommendation system, for exemple to recommend texts, like recent news articles that are appropriate for someone’s language level. <br>
There 6 different language levels: 
* A1 : begginer 
* A2 : elementary
* B1 : intermediate
* B2 : upper intermediate 
* C1 : advanced
* C2 : native

## 2. Methodology 
In this project we have tried several algorithms. <br>
First, we tried several **text cleaning and pre-processing** algorithms like : **tokenization** and **stop-words removal**. <br>
Then we have tried a **logistic regression** model and we have also tried some text classification techniques like: **k-nearest neighbor**, **decision tree** and **random forest**. <br>
Finaly, we decided to apply **BERT multilingual model**, which gave us the best results in accuracy of our predictions. <br> 

### Tokenization 
[Tokenization](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#tokenization) is the process of breaking down a stream of text into words, phrases, symbols, or any other meaningful elements called tokens. The main goal of this step is to extract individual words in a sentence. 

### Stop-words
A [stop word](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/) is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query. 

### Logistic Regression
[Logistic regression](https://www.javatpoint.com/logistic-regression-in-machine-learning) is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.

### K-nearest neighbor 
In machine learning, the [k-nearest neighbors algorithm (kNN)](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#k-nearest-neighbor) is a non-parametric technique used for classification. This method is used in Natural-language processing (NLP) as a text classification technique in many researches in the past decades.

### Decision Tree
One of earlier classification algorithm for text and data mining is [decision tree](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#decision-tree). Decision tree classifiers (DTC's) are used successfully in many diverse areas of classification. The structure of this technique includes a hierarchical decomposition of the data space (only train dataset). Decision tree as classification task was introduced by D. Morgan and developed by JR. Quinlan. The main idea is creating trees based on the attributes of the data points, but the challenge is determining which attribute should be in parent level and which one should be in child level. To solve this problem, De Mantaras introduced statistical modeling for feature selection in tree.

### Random Forests
[Random forests](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#random-forest) or random decision forests technique is an ensemble learning method for text classification. This method was introduced by T. Kam Ho in 1995 for first time which used t trees in parallel. This technique was later developed by L. Breiman in 1999 that they found converged for RF as a margin measure.

### BERT multilingual
[BERT](https://huggingface.co/bert-base-multilingual-cased) is a transformers model pretrained on a large corpus of multilingual data in a self-supervised fashion.

## 3. Results 

### Log 3 : Final Results
..... <br>
We have obtained the following results: 
|           | BERT multilingual | 
| ----------| ----------------- | 
| Precision |                   | 
| Recall    |                   | 
| F1 score  |                   |
| Accuracy  |      0.55583      |

### Log 2
In this log we trained originally given data on our 4 models, which we have built in our Log 1. 

We have obtained the following results: 
|           | Logistic Regression | kNN   | Decision Tree | Random Forests |
| ----------| ------------------- | ----- | ------------- | -------------  |
| Precision |         0.45        | 0.38  |      0.31     |      0.40      |
| Recall    |         0.45        | 0.31  |      0.32     |      0.40      |
| F1 score  |         0.45        | 0.29  |      0.30     |      0.39      |
| Accuracy  |         0.45        | 0.31  |      0.32     |      0.40      |

### Log 1
In our first log we have removed stop-words from the data and then we trained our models on this data. <br>
We have built 4 models: <br>
* Logistic Regression
* kNN
* Decision Tree
* Random Forests 

We have obtained the following results: 
|           | Logistic Regression | kNN   | Decision Tree | Random Forests |
| ----------| ------------------- | ----- | ------------- | -------------  |
| Precision |         0.40        | 0.11  |      0.42     |      0.41      |
| Recall    |         0.40        | 0.17  |      0.24     |      0.39      |
| F1 score  |         0.39        | 0.06  |      0.20     |      0.38      |
| Accuracy  |         0.40        | 0.17  |      0.24     |      0.39      |
