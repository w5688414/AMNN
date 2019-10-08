from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os


   
def save_image(text,display=False,file_name="first_review.png"):
# Create and generate a word cloud image:
    wordcloud = WordCloud(background_color = "white").generate(text)
    
    wordcloud.to_file(file_name)
# Display the generated image:
    if(display):
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


# lower max_font_size
# wordcloud = WordCloud(max_font_size=40).generate(text)
# plt.figure()
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()

# Save the image in the img folder:


## nus-wide
save_file_name='nus-wide.png'
image_text_data='/home/eric/Documents/Hashtag-recommendation-for-social-images/neural_image_captioning/datasets/NUS-WIDE/datasets.txt'
# Read the whole text.
with open(image_text_data,'r') as f:
    examples=f.readlines()
text=""
for example in examples:
    hashtags=example.strip().split('*')[1]
    text+=" "+hashtags
save_image(text,file_name=save_file_name)
# print(text)


## custom instagram
save_file_name='custom_instagram.png'
image_text_data='/home/eric/Documents/Hashtag-recommendation-for-social-images/image_text_hashtagging/datasets/image_text/image_text_data.txt'
# Read the whole text.
with open(image_text_data,'r') as f:
    examples=f.readlines()
text=""
for example in examples:
    hashtags=example.strip().split('*')[2]
    text+=" "+hashtags
save_image(text,file_name=save_file_name)


save_file_name='harrison.png'
image_text_data='/home/eric/Documents/Hashtag-recommendation-for-social-images/neural_image_captioning/datasets/HARRISON/image_tags.txt'
# Read the whole text.
with open(image_text_data,'r') as f:
    examples=f.readlines()
text=""
for example in examples:
    hashtags=example.strip().split('*')[1]
    text+=" "+hashtags
save_image(text,file_name=save_file_name)


