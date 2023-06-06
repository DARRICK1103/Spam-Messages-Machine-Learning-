import os
import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

# import data set into prototype
spam = pd.read_csv("SMSSpamDataCollection.csv", header=0,
                   names=['Type', 'Text'], encoding= 'unicode_escape')
print(f'The size of data set: {spam.shape}')# show the size of data set
print('\nHere is the list of data set: ')
print(spam)

# creating a new Type column by using data frame, with 0 for ham, 1 for spam
df = spam.drop(columns="Type")
df["spam"] = spam["Type"].apply(lambda x: 1 if 'spam' in str(x) else 0)
df.columns = ['Text', 'Type']
print('\nHere is the list of data set after feature engineering: ')
print(df)

# calculate the percentage of spam data in data set
print(f"\nPercentage of spam message: {df['Type'].mean()*100}")

# create two df which are 1 for spam and 1 for ham messages
df_s = df.loc[df['Type'] == 1]
df_h = df.loc[df['Type'] == 0]

# calculate the length of spam text in Text column and find out the average length
df_s['TLength'] = [len(x) for x in df_s["Text"]]
spamavg = df_s.TLength.mean()

# calculate the length of ham text in Text column and find out the average length
df_h['TLength'] = [len(x) for x in df_h["Text"]]
hamavg = df_h.TLength.mean()

# print out the average length of the messages
print(f'\nThe average length of spam messages is {spamavg}')
print(f'The average length of ham messages is {hamavg}')

# calculate the number of numbers in text
df['NLength'] = df['Text'].apply(lambda x: len(''.join([a for a in x if a.isdigit()])))
# show the mean number of numbers in ham message and the mean number of numbers in spam message
print('The mean number of numbers in message: ')
print('\t\tHam', "\t\t\t\tSpam")
print(np.mean(df['NLength'][df['Type'] == 0]), np.mean(df['NLength'][df['Type'] == 1]))
# show the list of the data set with number of numbers in the text
print(df)

# train test split
# split the data set for training
X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Type'], random_state=0)

# defining an additional function for after usage
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# fit and transfer X_train and X_test
vectorizer = TfidfVectorizer(min_df=5)

# find out how often a word appear in text column for X_train and X_test
X_train_transformed = vectorizer.fit_transform(X_train)
X_train_transformed_with_length = add_feature(X_train_transformed, X_train.str.len())

X_test_transformed = vectorizer.transform(X_test)
X_test_transformed_with_length = add_feature(X_test_transformed, X_test.str.len())

# SVM creation, set C as larger as possible to reduce misclassified
clf = SVC(C=10000)
print(f'\nMethod: {clf.fit(X_train_transformed_with_length, y_train)}')

# calculate the area under ROC curve
y_predicted = clf.predict(X_test_transformed_with_length)
print(f'Area under ROC curve: {roc_auc_score(y_test, y_predicted)}\n')

# confusion matrix
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
# show the confusion matrix
print(pd.DataFrame(confusion_matrix(y_test, y_predicted), columns=['Predicted Spam', "Predicted Ham"],
                   index=['Actual Spam', 'Actual Ham']))
# show the values from confusion matrix
print(f'\nTrue Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')
# calculate result from the values
print(f'\nTrue Positive Rate: { (tp / (tp + fn))}')
print(f'Specificity: { (tn / (tn + fp))}')
print(f'False Positive Rate: { (fp / (fp + tn))}')
