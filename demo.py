import tensorflow as tf
from tensorflow import keras
tf.__version__
import os
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Conv1D,MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample, shuffle

num_classes=2
data = pd.read_csv("../datasets/creditcard-trainingsetv2.csv")

#Preprocess Data
data.loc[data['Fraud? (1: Fraud, 0:  No Fraud)']==1,'Fraud']=1
data.loc[data['Fraud? (1: Fraud, 0:  No Fraud)']==0,'Fraud']=0
data.loc[data['Fraud? (1: Fraud, 0:  No Fraud)']==0,'Normal']=1
data.loc[data['Fraud? (1: Fraud, 0:  No Fraud)']==1,'Normal']=0
data['normAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data=data.drop(['Fraud? (1: Fraud, 0:  No Fraud)','Seconds since reference time','Amount'],axis=1)
fraud=data[data.Fraud==1]
normal=data[data.Fraud==0]
normal=resample(normal,replace=False,n_samples=417)
x_train=fraud.sample(frac=0.8)
x_train=pd.concat([x_train,normal.sample(frac=0.8)],axis=0)
x_test=data.loc[~data.index.isin(x_train.index)]
x_train=shuffle(x_train)
x_test=shuffle(x_test)

#Labels
y_train=x_train.Fraud
y_train=pd.concat([y_train,x_train.Normal],axis=1)
y_test=x_test.Fraud
y_test=pd.concat([y_test,x_test.Normal],axis=1)

#Inputs
x_train=x_train.drop(["Fraud","Normal"],axis=1)
x_test=x_test.drop(["Fraud","Normal"],axis=1)

#Hyperparameters
learning_rate=1e-7

# Train the model, iterating on the data in batches
inputs=Input(shape=(30,))
encoder = Dense(30, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(inputs)
encoder = Dense(14, activation="relu")(encoder)
decoder = Dense(14, activation='tanh')(encoder)
decoder = Dense(30, activation='relu')(decoder)
decoder = Dense(2,activation='softmax')(decoder)
model=Model(inputs,decoder)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
filepath = "model_{epoch:02d}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [checkpoint]
model.fit(x_train,y_train,epochs=100,batch_size=128,callbacks=callbacks_list)

#Evaluate Model
modelX=load_model('model_500.h5')
test2=np.random.random((1,30))
results=modelX.predict(x_test[:20])
print('results',results)
