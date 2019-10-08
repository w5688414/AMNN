from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
import time
from scipy import sparse
import gc
import os
from tqdm import tqdm
root_path='/home/eric/data/Mediamill'
f = open(os.path.join(root_path,'Mediamill_data.txt'),
'r')


size = f.readline()
nrows, nfeature,nlabel = [int(s) for s in size.split()]
x_m = [[] for i in range(nrows)]
pos = [[] for i in range(nrows)]
y_m = [[] for i in range(nrows)]

for i in tqdm(range(nrows)):
    line = f.readline()
    temp=[s for s in line.split(sep=' ')]
    pos[i]=[int(s.split(':')[0]) for s in temp[1:]]
    x_m[i]=[float(s.split(':')[1]) for s in temp[1:]]
    for s in temp[0].split(','):
        try:
            int(s)
            y_m[i]=[ int(s) for s in temp[0].split(',')]
        except:
            y_m[i]=[]



f = open(os.path.join(root_path,'mediamill_trSplit.txt'),'r')
#f = open(r'C:\Users\kaoyuant\Downloads\Mediamill\Mediamill_trSplit.txt',
#         'r',encoding='utf-8')
train=f.readlines()

f = open(os.path.join(root_path,'mediamill_tstSplit.txt'),'r')
#f = open(r'C:\Users\kaoyuant\Downloads\Mediamill\Mediamill_tstSplit.txt',
#         'r',encoding='utf-8')
test=f.readlines()


select=0
train_=[int(s.split()[select])-1 for s in train]
test_=[int(s.split()[select])-1 for s in test]

xm_train=[x_m[i] for i in train_]
ym_train=[y_m[i] for i in train_]

xm_test=[x_m[i] for i in test_]
ym_test=[y_m[i] for i in test_]



x_train=sparse.lil_matrix((len(train_),nfeature))
for i in tqdm(range(len(train_))):
    for j in range(len(pos[i])):
        x_train[i,pos[i][j]]=xm_train[i][j]

x_test=sparse.lil_matrix((len(test_),nfeature))
for i in tqdm(range(len(test_))):
    for j in range(len(pos[i])):
        x_test[i,pos[i][j]]=xm_test[i][j]

del x_m, xm_train, pos,xm_test
gc.collect()

y_train=sparse.lil_matrix((len(train_),nlabel))
for i in tqdm(range(len(train_))):
    for j in ym_train[i]:
        y_train[i,j]=1

y_test=sparse.lil_matrix((len(test_),nlabel))
for i in tqdm(range(len(test_))):
    for j in ym_test[i]:
        y_test[i,j]=1

del y_m, ym_train, ym_test
gc.collect()  


parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
score = 'f1_micro'

start=time.time()

# classifier = GridSearchCV(MLkNN(), parameters, scoring=score)
classifier = MLkNN(k=5)
classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)
print(predictions)
# print('training time taken: ',round(time.time()-start,0),'seconds')
# print('best parameters :', classifier.best_params_, 'best score: ',
#       classifier.best_score_)



#f = open(r'C:\Users\kaoyuant\Downloads\Mediamill\Mediamill_data.txt',
#         'r',encoding='utf-8')


