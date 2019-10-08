import pandas as pd
import numpy as np
from tqdm import tqdm


def getAverageHashtags(hashtags):
    tags=set()
    count=0
    max_length=0
    min_length=9999
    num_examples=hashtags.shape[0]
    for i in range(num_examples):
        text_arr=hashtags[i].split()
        text_len=len(text_arr)
        count+=text_len
        if(text_len==175):
            # print(text_arr)
            print(hashtags[i])
        max_length=max(max_length,text_len)
        
        min_length=min(min_length,text_len)
        for item in text_arr:
            tags.add(item)

    print("count:"+str(count))
    print("number of unique words: {}".format(len(tags)))
    print("max length: {}".format(max_length))
    print("min length: {}".format(min_length))
    print("average per image: {}".format(count/num_examples))

data_filename='../datasets/image_text/image_text_data.txt'
data = pd.read_table(data_filename, sep='*')
data = np.asarray(data)
captions = data[:, 1]
getAverageHashtags(captions)