# DMML 2022 - Detecting the difficulty level of French text
<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Logo_Université_de_Lausanne.svg/1280px-Logo_Université_de_Lausanne.svg.png" width="250" height="100"/> <br>
 </p>

**Team** : Zurich <br>
**Team members** : Emiliya AKHMETOVA, Feryel BEN RHAIEM

##   
<p align="center">
<img width="556" alt="image" src="https://user-images.githubusercontent.com/114418712/209107688-b89bbaa1-27ac-4a65-9f2f-ef71fdfe516e.png">
</p>

#### This github repository contains the following folders:
* **Code** : includes 6 .py files for Exploratory Data Analysis (EDA), data preparation, 4 models and their functions and final BERT model and its functions. 
* **Data** : includes 3 .csv files: training data, unlabelled test data and enriched training data (for better results, new records were added to the existing training data to enlarge our dataset) 
* **Final Submission** : includes 1 .csv file with ids of the unlabeled french sentences and predicted language difficulty level for final submission on [kaggle](https://www.kaggle.com/competitions/detecting-french-texts-difficulty-level-2022/overview) 
* **Results - 4 First Models** : includes 4 .png files, each one has a table for recall, precision, f1-score and accuracy results and a confusion matrix for our 4 models: logistic regression, kNN, random forest and decision tree
* **Slides Recap** : includes important imformation from the course which helped us in bulding our models

## 🚧  Project Description  
As a part of the Data Mining and Machine Learning course at the MScIS program in the University of Lausanne we constructed this model to predict the difficulty level of french text. The utility of a such model could be used in a recommendation system, for exemple to recommend text, like recent news articles that are appropriate for someone’s language level. <br>
There 6 different CEFR French Levels: 
* A1 : Beginner 
* A2 : Elementary
* B1 : Intermediate
* B2 : Upper intermediate 
* C1 : Advanced
* C2 : Native

## 🤔  Methodology 
In this project we have tried several algorithms. <br>
First, we tried several **text cleaning and pre-processing** algorithms like : **tokenization** and **stop-words removal**. <br>
Then we have implemented a **logistic regression** model, **k-nearest neighbor**, **decision tree** and **random forest**. <br>
Finally, we applied **BERT multilingual model**, which gave us the best results in accuracy of the predictions. <br> 

### Tokenization 
[Tokenization](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#tokenization) is the process of breaking down a stream of text into words, phrases, symbols, or any other meaningful elements called tokens. The main goal of this step is to extract individual words in a sentence. In the first 4 models, we vectorized our text using TfidfVectorizer(). 

### Logistic Regression
[Logistic regression](https://www.javatpoint.com/logistic-regression-in-machine-learning) is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.

### k-Nearest Neighbor 
In machine learning, the [k-Nearest Neighbors algorithm (kNN)](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#k-nearest-neighbor) is a non-parametric technique used for classification. This method is used in Natural-language processing (NLP) as a text classification technique in many researches in the past decades.

### Decision Tree
One of earlier classification algorithm for text and data mining is [decision tree](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#decision-tree). Decision tree classifiers (DTC's) are used successfully in many diverse areas of classification. The structure of this technique includes a hierarchical decomposition of the data space (only train dataset). Decision tree as classification task was introduced by D. Morgan and developed by JR. Quinlan. The main idea is creating trees based on the attributes of the data points, but the challenge is determining which attribute should be in parent level and which one should be in child level. To solve this problem, De Mantaras introduced statistical modeling for feature selection in tree.

### Random Forests
[Random forests](https://github.com/kk7nc/Text_Classification/blob/master/README.rst#random-forest) or random decision forests technique is an ensemble learning method for text classification. This method was introduced by T. Kam Ho in 1995 for first time which used t trees in parallel. This technique was later developed by L. Breiman in 1999 that they found converged for RF as a margin measure.

### BERT multilingual
[BERT](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2) is a model pretrained on a large corpus of multilingual data in a self-supervised fashion.

## 🎯  Implementation & Results 

### Take 1 🎬
In our first try we have removed stop-words from the data and then we trained our models on this data. <br>
We have built 4 models: <br>
* Logistic Regression
* kNN
* Decision Tree
* Random Forests 

We have obtained the following results: 
|           | Logistic Regression | kNN   | Decision Tree |  Random Forest |
| ----------| ------------------- | ----- | ------------- | -------------  |
| Precision |         0.40        | 0.11  |      0.42     |      0.41      |
| Recall    |         0.40        | 0.17  |      0.24     |      0.39      |
| F1 score  |         0.39        | 0.06  |      0.20     |      0.38      |
| Accuracy  |         0.40        | 0.17  |      0.24     |      0.39      |


### Take 2 🎬
In this try we trained the training data on our 4 models without data cleaning. We used a pipeline to vectorize the data using [TfidfVectorizer()](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) and then applied each model. 

#### Fine Tuning
In this step, we tried different parameters in the different models to finally choose the ones that gave us the best results.

We have obtained the following results: 
|           | Logistic Regression | kNN   | Decision Tree |  Random Forest |
| ----------| ------------------- | ----- | ------------- | -------------  |
| Precision |         0.46        | 0.38  |      0.29     |      0.39      |
| Recall    |         0.46        | 0.32  |      0.29     |      0.39      |
| F1 score  |         0.46        | 0.29  |      0.28     |      0.38      |
| Accuracy  |         0.46        | 0.32  |      0.29     |      0.39      |


### Take 3 🎬 Final Results - BERT Multilingual

#### How does BERT work?
[BERT](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2) (Bidirectional Encoder Representations from Transformers) is a Machine Learning model based on transformers, i.e. attention components able to learn contextual relations between words.

We used the [universal-sentence-encoder-cmlm/multilingual-base](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1) model, a universal sentence encoder that supports more than 100 languages. It is trained using a conditional masked language model.

We turned our text into high-dimensional vectors that capture sentence-level semantics. We loaded the preprocessor and the encoder layers from the endpoints provided by [TensorFlow Hub](https://www.tensorflow.org/hub), and defined a function **get_embeddings()** to get the embeddings from input text.

#### More data - how?
Well... 🤖

We used [ChatGPT](https://openai.com/blog/chatgpt/) to generate about 30 additional sentences for each French level. As a result, **the accuracy went up from 53% to 55%.**

These are the final results obtained: 
|           | BERT Multilingual | 
| ----------| ----------------- | 
| Accuracy  |     0.55583 🥳    |

## ▶️  Video  
[![IMAGE ALT TEXT HERE](https://www.google.com/url?sa=i&url=https%3A%2F%2Fblog.fatquartershop.com%2Fyoutube-channel-membership%2F&psig=AOvVaw0PH5cXaVl54PqbVORs81Eu&ust=1671805728227000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCJCTq-e3jfwCFQAAAAAdAAAAABAE)](https://youtu.be/9kISHGS8tyQ)

## 🔗  If you want to run the project on Google Colab 
Click 👉 [here](https://drive.google.com/drive/folders/1kaYZfzzkylGjUnu4WIGjgQWFR3UUZlCE?usp=sharing) to access the Google Drive folder. 

## ✅  Voilà!  
![Alt Text](https://media.giphy.com/media/lD76yTC5zxZPG/giphy.gif?cid=ecf05e47pkj3l2rpocnzkqvb15hb5fs9rg5auaw5b5ge7nek&rid=giphy.gif&ct=g)
