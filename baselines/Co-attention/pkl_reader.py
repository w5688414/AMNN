import pickle

data_file='/home/eric/Documents/Hashtag-recommendation-for-social-images/baselines/data/preprocessed_data/id_to_word.p'
# f = open('dict_word.pkl', 'rb')
# for line in f:
#     print(line)
word = pickle.load(open(data_file, 'rb'), encoding='utf-8')
print(word)
