# -*- coding: utf-8 -*-
"""DataPreparation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13EvtQe2cT5ArYjVm7xzXf_rItxD3wovM
"""

# This class is to prepare our data for analysis; loading it, splitting it into train and test, etc.

# Import required packages 
from sklearn.model_selection import train_test_split
import pandas as pd 
import tensorflow as tf

# Loading data
df_train = pd.read_csv('training_data.csv')
df = pd.read_csv('training-data_enriched.csv') 
df_pred = pd.read_csv('unlabelled_test_data.csv')

# Encoding labels
def map_difficulty(df):
    df['difficulty_num'] = df.difficulty.map({
          'A1': 1,
          'A2': 2,
          'B1': 3,
          'B2': 4,
          'C1': 5,
          'C2': 6,
    })

map_difficulty(df)
map_difficulty(df_train)

# Splitting data for the 4 first models
X_train, X_test, y_train, y_test = train_test_split(
    df_train.sentence,     
    df_train.difficulty_num, 
    test_size=0.2,   
    random_state=0, 
    stratify = df_train.difficulty_num
)

# X and y split

X = df_train['sentence'] 
y = df_train['difficulty']

# Splitting data for BERT
y_ = tf.keras.utils.to_categorical(df["difficulty_num"].values)
x_train, x_test, y__train, y__test = train_test_split(df['sentence'], y_, test_size=0.25)