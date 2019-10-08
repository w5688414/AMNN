# -*- coding: utf-8 -*-
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
# X_train = np.array(["new york is a hell of a town",
#                     "new york was originally dutch",
#                     "the big apple is great",
#                     "new york is also called the big apple",
#                     "nyc is nice",
#                     "people abbreviate new york city as nyc",
#                     "the capital of great britain is london",
#                     "london is in the uk",
#                     "london is in england",
#                     "london is in great britain",
#                     "it rains a lot in london",
#                     "london hosts the british museum",
#                     "new york is great and so is london",
#                     "i like london better than new york"])
# y_train_text = [["new york"],["new york"],["new york"],["new york"],["new york"],
#                 ["new york"],["london"],["london"],["london"],["london"],
#                 ["london"],["london"],["new york","london"],["new york","london"]]

# X_test = np.array(['nice day in nyc',
#                    'welcome to london',
#                    'london is rainy',
#                    'it is raining in britian',
#                    'it is raining in britian and the big apple',
#                    'it is raining in britian and nyc',
#                    'hello welcome to new york. enjoy it here and london too'])
# target_names = ['New York', 'London']

data_path='./data/'
training_filename = data_path + 'training_data.txt'
print('Loading training dataset...')
train_data = pd.read_csv(training_filename, delimiter='*')
train_data.drop(columns=['image_names'],inplace=True)
# train_data = train_data.values.tolist()
print(len(train_data))
x_train=train_data['tweets'].values
y_train=train_data['hashtags'].apply(lambda x:x.split()).values
print(x_train[0])
print(y_train[0])

validation_filename = data_path + 'validation_data.txt'
validation_data = pd.read_csv(validation_filename, delimiter='*')
validation_data.drop(columns=['image_names'],inplace=True)
# validation_data = validation_data.values.tolist()
print(len(validation_data))
X_test=validation_data['tweets'].values
Y_test=validation_data['hashtags'].values
print(X_test[0])
print(Y_test[0])


mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train)

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(x_train, Y)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

# for item, labels in zip(X_test, all_labels):
#     print('{0} => {1}'.format(item, ', '.join(labels)))

with open('svm_predict.txt','w') as f:
    for i in range(len(all_labels)):
        hashtags=[]
        for item in all_labels[i]:
            hashtags.append(item)
        f.write(' '.join(hashtags))
        f.write('\n')