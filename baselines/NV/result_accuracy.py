import numpy as np
import os
import pandas as pd
# root_path='/home/eric/Documents/Hashtag-recommendation-for-social-images/neural_image_captioning/datasets/Custom/preprocessed_data'
# data_path='./data/custom/'
# validation_filename = data_path + 'validation_data.txt'
# validation_data = pd.read_table(validation_filename, delimiter='*')
# validation_data.drop(columns=['tweets'],inplace=True)
# validation_data = validation_data.values.tolist()

data_path='./data/HARRISON/'
# data_path='./data/NUS-WIDE/'
validation_filename = data_path + 'validation_data.txt'
validation_data = pd.read_table(validation_filename, delimiter='*')
validation_data = validation_data.values.tolist()

print(validation_data[0])

dict_validation={}
for key,value in validation_data:
    dict_validation[key]=value


test_y=[]
pred_y=[]
pred_path=os.path.join(data_path,"predicted_hashtags.txt")
with open(pred_path,"r") as file:
    predicted_data=file.readlines()
    print(len(predicted_data))
    for i in range(len(predicted_data)):
        line_arr=predicted_data[i].strip().split("*")
        image_name=line_arr[0]
        target_label=dict_validation[image_name]
        target_arr=target_label.strip().split(' ')
        label_arr=line_arr[1].strip().split(' ')
#         label_arr = list(set(label_arr_ori)) 
#         label_arr.sort(key=label_arr_ori.index)
        # print(label_arr)
        # print(target_arr)
        test_y.append(target_arr)
        pred_y.append(label_arr)
# with open("target_captions.txt","r") as file:
#     target_labels=file.readlines()
    # print(target_labels)
print(test_y[:5])
print(pred_y[:5])
print(len(test_y))
print(len(pred_y))

def precision_score(test_y, pred_y, k=1):
    p_score = []
  
    for i in range(len(test_y)):
        # print(pred_y[i][-k:])
        # print(pred_y[i][-k])
        # result_at_topk = pred_y[i][-k:]
        count = 0
        # if(k>len(pred_y[i])):
        #     print(pred_y[i])
        end=min(k,len(pred_y[i]))
        for j in range(0,end):
            if(pred_y[i][j] in test_y[i]):
                count+=1
        p_score.append(float(count) / float(k))
            # if j in test_y[i]:
                # count += 1
        # p_score.append(float(count) / float(k))

    return np.mean(p_score)

def recall_score(test_y, pred_y, k=1):
    r_score = []
    for i in range(len(test_y)):
        count = 0
        end=min(k,len(pred_y[i]))
        for j in range(0,end):
            if(pred_y[i][j] in test_y[i]):
                count+=1
        r_score.append(float(count) / float(len(test_y[i])))

    return np.mean(r_score)

def hits_score(test_y, pred_y, k=1):
    h_score = []
    for i in range(len(test_y)):
        # result_at_topk = pred_y[i][-k:]
        # count = 0
        # for j in result_at_topk:
        #     if j in test_y[i]:
        #         count += 1
        count = 0
        end=min(k,len(pred_y[i]))
        for j in range(0,end):
            if(pred_y[i][j] in test_y[i]):
                count+=1
        h_score.append(1 if count > 0 else 0)

    return np.mean(h_score)

precisions=[]
recalls=[]
hits_rates=[]
num=5
names=[]
for i in range(num*3):
    if(i<num):
        names.append("precision@%d" %(i%num+1))
    elif(i<num*2):
        names.append("recall@%d" %(i%num+1))
    else:
        names.append("accuracy@%d" %(i%num+1))


    
for i in range(num):
    topk=i+1
    precision=precision_score(test_y,pred_y,topk)
    recall=recall_score(test_y,pred_y,topk)
    hits_rate=hits_score(test_y,pred_y,topk)
    print("precision@%d:%f" %(topk,precision))
    print("recall@%d:%f" %(topk,recall))
    print("accuracy@%d:%f" %(topk,hits_rate))
    precisions.append([precision])
    recalls.append([recall])
    hits_rates.append([hits_rate])
list_data=[]
list_data.extend(precisions)
list_data.extend(recalls)
list_data.extend(hits_rates)
print(list_data)
dict_data={}
for i in range(num*3):
    dict_data[names[i]]=list_data[i]

print(names)
result_at_topk=pd.DataFrame(data=dict_data)
result_at_topk.to_csv("result.csv")