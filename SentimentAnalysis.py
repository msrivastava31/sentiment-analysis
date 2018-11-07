#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as npy
import pandas as pd
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

#DATA PRE PROCESSING
def generate_review_label_array(path, review, label):
    with open(path, "r" ) as file:
        data = file.readlines()
        for line in data:
            temp_review = []

            words = line.split()
            #print(words)
            for word in words:
                element = word.split(":")
                # append multiple & handle last
                if element[0] != '#label#':
                    for i in range(0, int(element[1])):
                        temp_review.append(element[0])
                else:
                   label.append(element[1])
            review.append(" ".join(temp_review))

    print("Review Array : ",review[-1])
    print("Review Label : ",label[-1])
    print(len(review))
    print(len(label))
review = []
label = []
generate_review_label_array("/Users/medhasrivastava/Desktop/UW/CSS581/Project/processed_acl/books/positive.review",review, label)
generate_review_label_array("/Users/medhasrivastava/Desktop/UW/CSS581/Project/processed_acl/books/negative.review",review, label)
            


# In[111]:


#VECTORISATION
vectoriser = CountVectorizer()
transformer = vectoriser.fit_transform(review)
print("number words in training corpus:", len(vectoriser.get_feature_names()))
len(vectoriser.vocabulary_)
#print(vectoriser.toarray())


# In[112]:


transformer_25 = vectoriser.transform([review[24]])
print(transformer_25)
print(vectoriser.get_feature_names()[180290])


# In[121]:


review_nb = vectoriser.transform(review)
print('Shape of Sparse Matrix: ', review_nb.shape)
print('Amount of Non-Zero occurrences: ', review_nb.nnz)
# Percentage of non-zero values
density = (100.0 * review_nb.nnz / (review_nb.shape[0] * review_nb.shape[1]))
#print(‘Density: {}’.format((density)))
print("Density: ",density)


# In[123]:


#TRAINING AND TEST DATA : SPLIT
review_nb_train, review_nb_test, label_nb_train, label_nb_test = train_test_split(review_nb, label, test_size=0.2, random_state=101)


# In[124]:


#TRAINING OUR MODEL : MULTINOMIAL NAIVE BAYES
nb = MultinomialNB()
nb.fit(review_nb_train, label_nb_train)


# In[125]:


#TESTING & EVALUATING OUR MODEL
prediction = nb.predict(review_nb_test)
print(confusion_matrix(label_nb_test, prediction))
print('\n')
print(classification_report(label_nb_test, prediction))


# In[137]:


# calculate accuracy, precision, recall, and F-measure of class predictions
from sklearn import metrics
def eval_predictions(label_nb_test, prediction):
    print("accuracy:", metrics.accuracy_score(label_nb_test, prediction))
    print("precision:", metrics.precision_score(label_nb_test, prediction, average='weighted'))
    print("recall:", metrics.recall_score(label_nb_test, prediction, average='weighted'))
    print("F-measure:", metrics.f1_score(label_nb_test, prediction, average='weighted'))
eval_predictions(label_nb_test, prediction)


# In[149]:


#TRAINING OUR MODEL : LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

review_lr = vectoriser.transform(review)
review_lr_train, review_lr_test, label_lr_train, label_lr_test = train_test_split(
    review_lr, label, train_size = 0.8
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(review_lr_train, label_lr_train)
    print ("Accuracy for C : " , (c, accuracy_score(label_lr_test, lr.predict(review_lr_test))))
    
final_model = LogisticRegression(C=0.5)
final_model.fit(review_lr_train, label_lr_train)
print ("Final Accuracy : " , accuracy_score(label_lr_test, final_model.predict(review_lr_test)))
eval_predictions(label_lr_test,final_model.predict(review_lr_test))


# In[144]:


#SUPPORT VECTOR MACHINE
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

review_svm_train, review_svm_test, label_svm_train, label_svm_test = train_test_split(review, label, test_size=0.2, random_state=101)

#print(review_train[0])

tfidf_vectorizer_1 = TfidfVectorizer()
tfidf_train_1 = tfidf_vectorizer_1.fit_transform(review_svm_train)
tfidf_test_1 = tfidf_vectorizer_1.transform(review_svm_test)

# instantiate and train model, kernel=rbf 
svm_rbf = svm.SVC(random_state=1)
svm_rbf.fit(tfidf_train_1, label_svm_train)

# evaulate model
label_pred_rbf = svm_rbf.predict(tfidf_test_1)
eval_predictions(label_svm_test, label_pred_rbf)


# In[145]:


# instantiate and train model, kernel=linear
svm_linear = svm.SVC(kernel='linear', random_state=12345)
svm_linear.fit(tfidf_train_1, label_svm_train)

# evaulate model
label_pred_linear = svm_linear.predict(tfidf_test_1)
eval_predictions(label_svm_test, label_pred_linear)

