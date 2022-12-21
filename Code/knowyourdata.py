# -*- coding: utf-8 -*-
"""KnowYourData.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-pe7UyIxGs2AvHnoZsdFeUmlZKTe5yGo
"""

# Commented out IPython magic to ensure Python compatibility.
# Import packages 
# %matplotlib inline
import numpy as np
import tensorflow_hub as hub
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
# Import files 
import datapreparation 
from datapreparation import *

# Take a look at our training data
df_train

# Look at the distribution of the classes
classes = len(df_train["difficulty"].value_counts())

colors = plt.cm.Dark2(np.linspace(0, 1, classes))
iter_color = iter(colors)

df_train["difficulty"].value_counts().plot.barh(title="Difficulty Level for Each Sentence (n, %)", 
                                                 ylabel="Difficulty Levels",
                                                 color=colors,
                                                 figsize=(9,9))

for i, v in enumerate(df_train["difficulty"].value_counts()):
  c = next(iter_color)
  plt.text(v, i,
           " "+str(v)+", "+str(round(v*100/df_train.shape[0],2))+"%", 
           color=c, 
           va='center', 
           fontweight='bold')

preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1")

# A function that returns BERT-like embeddings of input text
def get_embeddings(sentences):
  '''return BERT-like embeddings of input text
  Args:
    - sentences: list of strings
  Output:
    - BERT-like embeddings: tf.Tensor of shape=(len(sentences), 768)
  '''
  preprocessed_text = preprocessor(sentences)
  return encoder(preprocessed_text)['pooled_output']

# Look at the semantic textual similarities
def plot_similarity(features, labels):
  cos_sim = cosine_similarity(features)
  fig = plt.figure(figsize=(10,8))
  sns.set(font_scale=1.2)
  cbar_kws=dict(use_gridspec=False, location="left")
  g = sns.heatmap(
      cos_sim, xticklabels=labels, yticklabels=labels,
      vmin=0, vmax=1, annot=True, cmap="Reds", 
      cbar_kws=cbar_kws)
  g.tick_params(labelright=True, labelleft=False)
  g.set_yticklabels(labels, rotation=0)
  g.set_title("Semantic Textual Similarity")

# Try it for 3 sentences from the training dataset
sentences = ["Le bleu, c'est ma couleur préférée mais je n'aime pas le vert!",
           "Est-ce que ton mari est aussi de Boston?",
           "Les médecins disent souvent qu'on doit boire un verre de vin rouge après les repas."]

plot_similarity(get_embeddings(sentences), sentences)

# Take a look at the enriched training data. Thanks ChatGPT! ;) 
df