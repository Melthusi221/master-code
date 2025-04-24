#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load in our libraries
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pandas.api.types import is_string_dtype
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels


# # PREPROCESSING

# In[2]:


dataset = pd.read_csv("C:/Users/HP/OneDrive/Documents/Dissertation/Master/Credit-Scoring-master/dataset/estadistical.csv")

# In[3]:


x = dataset.drop("Receive/ Not receive credit ", axis=1)
y = dataset["Receive/ Not receive credit "]


# In[4]:


cat_mask = x.dtypes == object
cat_cols = x.columns[cat_mask].tolist()


# In[5]:


le = preprocessing.LabelEncoder()
x[cat_cols] = x[cat_cols].apply(lambda col: le.fit_transform(col))


# In[6]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, stratify=y)

# Scale the data (BEST SOLUTION ADDED)
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)


# # KNN

# In[7]:


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(xtrain_scaled, ytrain)  # Using scaled data


# In[8]:


preKnn = neigh.predict(xtest_scaled)  # Using scaled data


# In[9]:


accuracy_score(ytest, preKnn)


# # RANDOM FOREST

# In[10]:


forest = RandomForestClassifier(max_depth=2, random_state=0)
forest.fit(xtrain_scaled, ytrain)  # Using scaled data


# In[11]:


pred_forest = forest.predict(xtest_scaled)  # Using scaled data


# In[12]:


accuracy_score(ytest, pred_forest)


# # LOGISTIC REGRESSION (OPTIMIZED)

# In[23]:


logRegr = LogisticRegression(
    solver='liblinear',  # Best solver for small/medium datasets
    max_iter=1000,      # Increased iterations
    random_state=0,
    class_weight="balanced"
).fit(xtrain_scaled, ytrain)  # Using scaled data


# In[24]:


pred_logReg = logRegr.predict(xtest_scaled)  # Using scaled data


# In[25]:


accuracy_score(ytest, pred_logReg)


# In[21]:


# Updated confusion matrix plotting
disp = ConfusionMatrixDisplay.from_predictions(
    ytest, 
    pred_logReg,
    display_labels=ytrain.unique(),
    cmap=plt.cm.Greys
)
disp.ax_.set_title('Confusion matrix')
plt.show()


# # SVM

# In[27]:


svm = SVC(gamma='auto', class_weight="balanced")
svm.fit(xtrain_scaled, ytrain)  # Using scaled data


# In[28]:


pred_svm = svm.predict(xtest_scaled)  # Using scaled data


# In[29]:


accuracy_score(ytest, pred_svm)


# # CONFUSION MATRIX (Function Definition)

# In[20]:


def plot_confusion_matrix(y_true, y_pred, classes,
                         normalize=False,
                         title=None,
                         cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center")
# KNN Accuracy (After In[9])
print(f"KNN Accuracy: {accuracy_score(ytest, preKnn):.2%}")

# Random Forest Accuracy (After In[12])
print(f"Random Forest Accuracy: {accuracy_score(ytest, pred_forest):.2%}")

# Logistic Regression Accuracy (After In[25])
print(f"Logistic Regression Accuracy: {accuracy_score(ytest, pred_logReg):.2%}")

# SVM Accuracy (After In[29])
print(f"SVM Accuracy: {accuracy_score(ytest, pred_svm):.2%}")            
print("amanda")                    