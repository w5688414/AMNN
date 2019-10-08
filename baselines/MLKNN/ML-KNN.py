
import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

data_path='./data/'
training_filename = data_path + 'training_data.txt'
print('Loading training dataset...')
train_data = pd.read_table(training_filename, delimiter='*')
train_data.drop(columns=['image_names'],inplace=True)
# train_data = train_data.values.tolist()
print(len(train_data))
x_train=train_data['tweets'].values
y_train=train_data['hashtags'].values


validation_filename = data_path + 'validation_data.txt'
validation_data = pd.read_table(validation_filename, delimiter='*')
validation_data.drop(columns=['image_names'],inplace=True)
# validation_data = validation_data.values.tolist()
# print(len(validation_data))
X_test=validation_data['tweets'].values
Y_test=validation_data['hashtags'].values



vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vectorizer.fit(x_train)
vectorizer.fit(X_test)
feature_name = vectorizer.get_feature_names()
print(feature_name)
pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))

x_train = vectorizer.transform(x_train)
x_test = vectorizer.transform(X_test)

word_to_id_filename='word_to_id.p'
id_to_word_filename='id_to_word.p'
word_to_id = pickle.load(open(data_path +
                                           word_to_id_filename, 'rb'))
id_to_word = pickle.load(open(data_path +
                                           id_to_word_filename, 'rb'))


max_len=1003
def to_category_vector(texts,max_len):
    vector = np.zeros(max_len).astype(np.float32)
    for word in texts :
        vector[word_to_id[word]]=1.0
    return vector
document_X = []
document_Y = []
test_y=[]
for example in tqdm(y_train):
    arr=example.strip().split()
    document_Y.append(to_category_vector(arr,max_len))
document_Y=np.array(document_Y)
for example in tqdm(Y_test):
    arr=example.strip().split()
    test_y.append(to_category_vector(arr,max_len))
test_y=np.array(test_y)

classifier_new = MLkNN(k=5)
classifier_new.fit(x_train, document_Y)
# predict
predictions_new = classifier_new.predict(x_test)
# print(predictions_new)

pred=predictions_new.toarray()
with open("data/predict.txt",'w') as f:
    for i in tqdm(range(pred.shape[0])):
        g=np.where(pred[i]==1)
        hashtags=[]
        for item in g[0]:
            word=id_to_word[item]
            hashtags.append(word)
        f.write(' '.join(hashtags))
        f.write('\n')