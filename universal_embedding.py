# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:17:11 2019

@author: BolluD
"""
#=============================================================================
# idenftifying the questions related to given labels 
#=============================================================================
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

import matplotlib as inline

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

print("TF Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("TF Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

#import tensorflow as tf
#import tensorflow_hub as hub
#module_url = (r'C:\Users\bollud\Downloads\3.tar.gz')
# Import the Universal Sentence Encoder's TF Hub module
#embed = hub.Module(module_url)
tf.compat.v1.disable_eager_execution()
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")

# Compute a representation for each message, showing various lengths supported.
messages = ["That band rocks!", "That song is really cool."]
  
with tf.compat.v1.Session() as session:
    session.run([tf.compat.v1.global_variables_initializer(), 
                 tf.compat.v1.tables_initializer()])
    message_embeddings = session.run(embed(messages))

message_embeddings, message_embeddings.shape

import re

def get_dataframe(filename):
    lines = open(filename, 'r').read().splitlines()
    data = []
    for i in range(0, len(lines)):
        label = lines[i].split(' ')[0]
        label = label.split(":")[0]
        text = ' '.join(lines[i].split(' ')[1:])
        text = re.sub('[^A-Za-z0-9 ,\?\'\"-._\+\!/\`@=;:]+', '', text)
        data.append([label, text])

    df = pd.DataFrame(data, columns=['label', 'text'])
    df.label = df.label.astype('category')
    return df

df_train = get_dataframe(r'C:\Users\bollud\Desktop\cd3.txt')
df_train.head()


train_text = df_train['text'].tolist()
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = np.asarray(pd.get_dummies(df_train.label), dtype = np.int8)

df_test = get_dataframe(r'C:\Users\bollud\Desktop\cd2.txt')
test_text = df_test['text'].tolist()
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = np.asarray(pd.get_dummies(df_test.label), dtype = np.int8)


def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
    	signature="default", as_dict=True)["default"]

import tensorflow as tf
from tensorflow import keras
from keras.layers import Embedding
category_counts = len(df_train.label.cat.categories)


input_text = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
embedding = tf.keras.layers.Lambda(UniversalEmbedding,output_shape=(512,))(input_text)
dense = tf.keras.layers.Dense(256, activation='relu')(embedding)
pred = tf.keras.layers.Dense(category_counts, activation='softmax')(dense)
model = tf.keras.models.Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', 
	optimizer='adam', metrics=['accuracy'])

model.summary()

  
EPOCHS=8
BATCH_SIZE=32
  
# Fit the model
with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())  
    session.run(tf.compat.v1.tables_initializer())
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                      patience=3,
                                      restore_best_weights=True,
                                      verbose=1)
    
    model.fit(train_text, train_label, 
              validation_data=(test_text, test_label),
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              shuffle=True,
              callbacks=[es],
              verbose=1)
    
    model.save_weights('./use8_model.h5')
    
    
new_text = ["In what year did the titanic sink ?", 
            "What is the highest peak in California ?", 
            "Who invented the light bulb ?",
            "Where is pacific ocean ?",
            "who invented computer ?",
            "what is the use of book ?"
            ]

new_text = np.array(new_text, dtype=object)[:, np.newaxis]
with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())  
    session.run(tf.compat.v1.tables_initializer())
    model.load_weights('./use8_model.h5')  
    predicts = model.predict(new_text, batch_size=32)

categories = df_train.label.cat.categories.tolist()
predict_logits = predicts.argmax(axis=1)
predict_labels = [categories[logit] for logit in predict_logits]
print(predict_labels)

#=============================================================================



#=============================================================================
# NEWS CLASSIFICATION 
#=============================================================================

news_data=pd.read_excel(r'C:\Users\bollud\sent.xlsx')
news_train=news_data["NEWS"]

train_news=news_train[:800].tolist()
train_news = np.array(train_news, dtype=object)[:, np.newaxis]

val_news=news_train[800:1000].tolist()
val_news = np.array(val_news, dtype=object)[:, np.newaxis]

news_test_data=pd.read_excel(r'C:\Users\bollud\sent2.xlsx')
news_test=news_test_data["NEWS"].tolist()
news_test = np.array(news_test, dtype=object)[:, np.newaxis]

from keras.utils import to_categorical
y=to_categorical(news_data['label'])
y_train=y[:800]
val_y=y[800:1000]
#news_train=news_train['NEWS'].apply(lambda x: x.lower()) 

tf.compat.v1.ensure_shape
tf.compat.v1.disable_eager_execution()
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
#tf.compat.v1.disable_eager_execution()
with tf.compat.v1.Session() as session:
    session.run([tf.compat.v1.global_variables_initializer(), 
                 tf.compat.v1.tables_initializer()])
    message_embeddings = session.run(embed(train_news[:3]))

message_embeddings, message_embeddings.shape


def UniversalEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), 
    	signature="default", as_dict=True)["default"]

input_text = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
embedding = tf.keras.layers.Lambda(UniversalEmbedding,output_shape=(512,))(input_text)
dense = tf.keras.layers.Dense(60, activation='relu')(embedding)
pred = tf.keras.layers.Dense(4, activation='softmax')(dense)
model = tf.keras.models.Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', 
	optimizer='adam', metrics=['accuracy'])

model.summary()

EPOCHS=10
BATCH_SIZE=32
  
# Fit the model
with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())  
    session.run(tf.compat.v1.tables_initializer())
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                      patience=3,
                                      restore_best_weights=True,
                                      verbose=1)
    
    model.fit(train_news, y_train, 
              validation_data=(val_news, val_y),
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              shuffle=True,
              callbacks=[es],
              verbose=1)
    
    model.save_weights('./sent_model.h5')
    
new_text = [ " will the movies educating the people or spoiling",
           "cricket is the one of top rated game among all the sports games",
           "mobiles are getting increased day by day and creating more radiations in environment"]

            #"What is the highest peak in California ?", 
           # "Who invented the light bulb ?",
            #"Where is pacific ocean ?",
           # "who invented computer ?",
           # "what is the use of book ?"
    

new_text = np.array(new_text, dtype=object)[:, np.newaxis]
with tf.compat.v1.Session() as session:
    session.run(tf.compat.v1.global_variables_initializer())  
    session.run(tf.compat.v1.tables_initializer())
    model.load_weights('./sent_model.h5')  
    predicts = model.predict(news_test, batch_size=32)

categories = [0,1,2,3]
predict_logits = predicts.argmax(axis=1)
predict_labels = [categories[logit] for logit in predict_logits]
print(predict_labels)


#df["whitespace_removed"].to_excel("sent.xlsx",index=false)
#submission= ({"NEWS": news_test_data["NEWS"],
 #             "predicted_label": predict_labels})
#pd.DataFrame(submission).to_excel("universal2.xlsx",index=False)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("Accuracy: %.2f%%" % (accuracy_score(news_test, predictions)*100))
print(classification_report(test_sentiments, predictions))
pd.DataFrame(confusion_matrix(test_sentiments, predictions))