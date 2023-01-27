#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Necessary Libraries and Functions

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px

from matplotlib.colors import ListedColormap

from scipy import stats
from scipy.stats import norm, skew

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# # **Reading The .csv File**

# In[2]:


# Read csv File
print('\n')
df = pd.read_csv(r"cancer patient data sets.csv", index_col='index')

# Display Data
display(df)
print('\n')


# # **Dropping and Cleaning Data**

# In[3]:


# Index Column now refers to patient

print('\n')
df.drop("Patient Id", axis=1, inplace=True)

# Cleaning Column Names
df.rename(columns=str.lower, inplace=True)
df.rename(columns={col: col.replace(" ", "_")
          for col in df.columns}, inplace=True)

# Display Data After Cleaning
display(df)
print('\n')


# # **Check for Null Values**

# In[4]:


# Check For Null Values

print('\n')
df.isnull().sum()


# # **Print Information**

# In[5]:


# Print Information

print('\n')
print(df.info())
print('\n')


# # **Replace "level" with Integer**

# In[6]:


# Replace "level" with Integer

print('\n')
print('Cancer Levels: ', df['level'].unique())

# Replacing levels with int
df["level"].replace({'High': 2, 'Medium': 1, 'Low': 0}, inplace=True)
print('Cancer Levels: ', df['level'].unique())

print('\nColumns in dataframe: \n', df.columns)
print('\n')


# In[7]:


# Round

print('\n')
round(df.describe().iloc[1:, ].T, 1)


# # **Print and Visualize Columns**

# In[8]:


# Print and Visualize Columns

print('\n')
df.columns


# In[9]:


# Consolidating Necessary Columns

cols = [
    'age', 'weight_loss', 'smoking'
]

cols2 = ['gender']

cols3 = [
    'air_pollution', 'alcohol_use', 'dust_allergy', 'smoking', 'chest_pain', 'fatigue'
]


# In[10]:


# Presenting the countplots for categorical features

print('\n')
for i in cols:
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    sns.countplot(y=df[i], data=df,
                  order=df[i].value_counts().index, palette='Blues_r')
    plt.ylabel(i)
    plt.yticks(fontsize=10)
    print("*************************************************************************************************************************")
    plt.box(False)
    plt.show()

# Gender
for i in cols2:
    fig, ax = plt.subplots(1, 1, figsize=(15, 2))
    sns.countplot(y=df[i], data=df,
                  order=df[i].value_counts().index, palette='Blues_r')
    plt.ylabel(i)
    plt.yticks(fontsize=8)
    print("**********************************************************************************************************************")
    plt.box(False)
    plt.show()

print('\n')


# In[11]:


# Histograms

print('\n')
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))
ax = ax.flatten()
i = 0

for c in cols3:
    df.hist(c, figsize=(4, 4), ax=ax[i], label=f'{c}')
    i = i + 1
plt.show()
print('\n')


# In[12]:


# Pie Chart

df['level'].value_counts().plot(kind='pie', figsize=(6, 6), autopct='%1.1f%%')
plt.show()


# In[13]:


# Scatter Plot

print('\n\tSmoking and Label of Lung Cancer')
plt.scatter(df['smoking'], df['level'])
plt.show()
print('\n')

print('\n\tAir Polution and Label of Lung Cancer')
plt.scatter(df['air_pollution'], df['level'])
plt.show()
print('\n')


# In[14]:


# Animated Scatter Plot

px.scatter(data_frame=df,
           x='age',
           y='smoking',
           size='air_pollution',
           color='level',
           title='Age, Smoking and Air Polution',
           labels={
               'age': 'Age',
               'air_pollution': 'Air Polution',
               'level': 'Lung Cancer Level',
               'smoking': 'Smoking',
               'gender': 'Gender'
           },
           log_x=True,
           range_y=[-5, 15],
           hover_name='level',
           animation_frame='gender',
           height=800,
           size_max=100
           )


# # **Heatmap**

# In[15]:


# Heatmap

print('\n')
plt.figure(figsize=(20, 15))
sns.heatmap(df.corr(), annot=True, cmap=plt.cm.PuBu)
plt.show()
print('\n')


# # **Setting Target**

# In[16]:


# Setting Target

X = df.drop(columns='level')
y = df.level

print('\n')
display(X.head(), y[:10])
print('\n')


# In[17]:


print('\n')
df.columns


# # **Train Test Split**

# In[18]:


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=40)
print('\nTrain Shape\n')
print('X train shape: ', X_train.shape)
print('Y train shape: ', y_train.shape)
print('\n\nTest Shape\n')
print('X test shape: ', X_test.shape)
print('Y test shape: ', y_test.shape)
print('\n')


# # **Scaling the Data**

# In[19]:


# Data Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # **Logistic Regression**

# In[20]:


# Logistic Regression

logreg = LogisticRegression(C=24)

logreg.fit(X_train_scaled, y_train)

y_predict1 = logreg.predict(X_test_scaled)


# **Confusion Matrix**

# In[21]:


# Confusion Matrix

logreg_cm = confusion_matrix(y_test, y_predict1)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(logreg_cm, annot=True, linewidth=0.7,
            linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('Logistic Regression Classification Confusion Matrix\n')
plt.xlabel('y predict')
plt.ylabel('y test')
print('\n')
plt.show()
print('\n')


# **Test Score**

# In[22]:


# Test Score

print('\n')
score_logreg = logreg.score(X_test_scaled, y_test)
print('Logistic Regression Score = ', score_logreg)
print('\n')


# In[23]:


# Classification Report

print('\nClassification Report for Logistic Regression\n')
print(classification_report(y_test, y_predict1))
print('\n')


# # **Gaussian Naive Bayes**

# In[24]:


# Gaussian Naive Bayes

nbcla = GaussianNB()

nbcla.fit(X_train_scaled, y_train)

y_predict2 = nbcla.predict(X_test_scaled)


# **Confusion Matrix**

# In[25]:


# Confusion Matrix

nbcla_cm = confusion_matrix(y_test, y_predict2)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(nbcla_cm, annot=True, linewidth=0.7,
            linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('Gaussian Naive Bayes Classification Confusion Matrix\n')
plt.xlabel('y predict')
plt.ylabel('y test')
print('\n')
plt.show()
print('\n')


# **Test Score**

# In[26]:


# Test Score

print('\n')
score_nbcla = nbcla.score(X_test_scaled, y_test)
print('Gaussian Naive Bayes Score = ', score_nbcla)
print('\n')


# In[27]:


# Classification Report

print('\nClassification Report for Gaussian Naive Bayes\n')
print(classification_report(y_test, y_predict2))
print('\n')


# # **Decision Tree**

# In[28]:


# Decision Tree

dtcla = DecisionTreeClassifier(random_state=9)

dtcla.fit(X_train_scaled, y_train)

y_predict3 = dtcla.predict(X_test_scaled)


# **Confusion Matrix**

# In[29]:


# Confusion Matrix

dtcla_cm = confusion_matrix(y_test, y_predict3)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(dtcla_cm, annot=True, linewidth=0.7,
            linecolor='cyan', fmt='g', ax=ax, cmap="YlGnBu")
plt.title('Decision Tree Classification Confusion Matrix\n')
plt.xlabel('y predict')
plt.ylabel('y test')
print('\n')
plt.show()
print('\n')


# **Test Score**

# In[30]:


# Test Score
print('\n')
score_dtcla = dtcla.score(X_test_scaled, y_test)
print('Decision Tree Score = ', score_dtcla)
print('\n')


# In[31]:


# Classification Report
print('\n\t\t\tClassification Tree for Decision Tree\n')
print(classification_report(y_test, y_predict3))
print('\n')


# # **All Test Scores**

# In[32]:


# All Test Scores

print('\n')
Testscores = pd.Series([score_logreg, score_nbcla, score_dtcla],
                       index=['Logistic Regression Score = ', 'Naive Bayes Score = ', 'Decision Tree Score = '])
print(Testscores)
print('\n')


# # **All Confusion Matrices**

# In[33]:


# All Confusion Matrices

print('\n')
fig = plt.figure(figsize=(20, 15))
ax1 = fig.add_subplot(3, 3, 1)
ax1.set_title('Logistic Regression Classification\n')
ax2 = fig.add_subplot(3, 3, 2)
ax2.set_title('Naive Bayes Classification\n')
ax3 = fig.add_subplot(3, 3, 3)
ax3.set_title('Decision Tree Classification\n')

sns.heatmap(data=logreg_cm, annot=True, linewidth=0.7,
            linecolor='cyan', cmap="YlGnBu", fmt='g', ax=ax1)
sns.heatmap(data=nbcla_cm, annot=True, linewidth=0.7,
            linecolor='cyan', cmap="YlGnBu", fmt='g', ax=ax2)
sns.heatmap(data=dtcla_cm, annot=True, linewidth=0.7,
            linecolor='cyan', cmap="YlGnBu", fmt='g', ax=ax3)

plt.show()
print('\n')


# # **Comparison**

# In[34]:


# Comparison of Algorithms

x = ['Logistic Regression', 'G. Naive Bayes', 'Decision Tree']
y = [score_logreg, score_nbcla, score_dtcla]


# In[35]:


# Bar Plot

print('\n')
plt.bar(x, y)
plt.xlabel('\nClassification Algorithms')
plt.ylabel("Scores\n")
plt.title('Classification Algorithms Score Comparison Bar Plot\n')
plt.show()
print('\n')


# In[36]:


# Scatter Plot

print('\n')
colors = np.random.rand(3)
plt.xlabel('\nClassification Algorithms')
plt.ylabel("Scores\n")
plt.title('Classification Algorithms Score Comparison Scatter Plot\n')
plt.scatter(x, y, s=200, c=colors)
plt.show()
print('\n')


# In[37]:


# Compare Scores and Find Out The Best Algorithm

al = False
ln = False
ld = False
nd = False

if score_logreg == score_nbcla and score_logreg == score_dtcla and score_nbcla == score_dtcla:
    al = True

if score_logreg == score_nbcla:
    ln = True

if score_logreg == score_dtcla:
    ld = True

if score_nbcla == score_dtcla:
    nd = True

if al:
    print('\nAll Models Perform The Same\n')
elif ln:
    print('\nLogistic Regression and Gaussian Naive Bayes Performs Better\n')
elif ld:
    print('\nLogistic Regression and Dicision Tree Performs Better\n')
elif nd:
    print('\nGaussian Naive Bayes and Decision Tree Performs Better\n')
else:
    if score_logreg > score_nbcla and score_logreg > score_dtcla:
        print('\nLogistic Regression Performs Better\n')
    if score_nbcla > score_logreg and score_nbcla > score_dtcla:
        print('\nGaussian Naive Bayes Performs Better\n')
    if score_dtcla > score_logreg and score_dtcla > score_nbcla:
        print('\nDicision Tree Performs Better\n')
