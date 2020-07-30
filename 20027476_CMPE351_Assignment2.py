#!/usr/bin/env python
# coding: utf-8

# Neural network assignment
# @author Stefan Robb

# In[70]:


import pandas
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np


# In[ ]:


dataframe = pandas.read_csv("recipes.csv")

X_columns = list(filter(lambda x: x != "Id" and x !=
                        "Cuisine", dataframe.columns))

X = dataframe.loc[:, X_columns].values
Y_unmodified = dataframe.loc[:, "Cuisine"].values
Y = np.array([[0] * 20 for _ in Y_unmodified])

# One hot encode the Y array
for i, y in enumerate(Y_unmodified):
    Y[i][y] = 1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

# Build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=len(X_columns)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(20, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=20)

Y_pred = model.predict(X_test)

# Modify so they work with SKL
Y_pred_modified = [np.argmax(y) for y in Y_pred]
y_test_modified = [np.argmax(y) for y in Y_test]

# 500 EPOCHS - 77%
# 200 EPOCHS - 78%
# 50 EPOCHS  - 78%
# 20 EPOCHS - 78%
# 10 EPOCHS - 79%
# 1 EPOCH - 75%
print(f"Accuracy: {accuracy_score(y_test_modified, Y_pred_modified)}")


# In[76]:


dataframe = pandas.read_csv(r"C:\Users\stefa\Desktop\Queens\Fourth Year\CMPE351\Project\Recipe Data\smallrecipes.csv", engine = 'python')  # input path to recipes csv


# In[ ]:


dataset = dataframe.values
X = dataset[:, 0:len(dataframe.columns) - 1].astype(int)
Y = dataset[:, len(dataframe.columns)].astype(int)


# In[ ]:


model = Sequential()
model.add(Dense(512, activation='relu', input_dim = len(dataframe.columns - 1)))
# model.add(Dense(6714, activation='relu', input_shape = (len(dataframe.columns - 1), ))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(20, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


estimator = KerasClassifier(build_fn=model, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

