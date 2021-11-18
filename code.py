import pandas as pnds
import matplotlib.pyplot as mplp
import nltk
import operator
nltk.download('punkt')

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string

from nltk.probability import FreqDist as FD


f1 = open(r"C:\Users\Khushboo\Downloads\T1.txt", "r",encoding="utf-8")
f2 = open(r"C:\Users\Khushboo\Downloads\T2.txt", "r",encoding="utf-8")
T1 = f1.read()
T2 = f2.read()

def text_preprocessor(text):
  # converting into lower case
  text= text.lower()
  # to remove links
  text = re.sub('https?://\S+|www\.\S+', '', text)
  # to remove punctuations
  text = re.sub('[^a-zA-Z]', '',text)
  # to remove running word
  text= re.sub("chapter [XVI]+","",text)
  return text


T1_words = T1.split()
print("number of words in T1:",len(T1_words))

T2_words = T2.split()
print("number of words in T2:",len(T2_words))

with open("T1.txt", 'r',encoding = "utf-8") as fp:
    num_lines = sum(1 for line in fp if line.rstrip())
    print('Total lines in T1:', num_lines)

with open("T2.txt", 'r',encoding = "utf-8") as fp:
    num_lines = sum(1 for line in fp if line.rstrip())
    print('Total lines in T2:', num_lines)


T1 = text_preprocessor(T1)
print(T1)

T2= text_preprocessor(T2)
print(T2)

token1=word_tokenize(T1)
print(token1)

token2=word_tokenize(T2)
print(token2)

fd1 = FD(token1)
print(fd1)

fd2 = FD(token2)
print(fd2)

K = 25
list_fd1 = dict(sorted(fd1.items(), key=operator.itemgetter(1),reverse=True))
list_fd2 = dict(sorted(fd2.items(), key=operator.itemgetter(1),reverse=True))

list_output_fd1 = dict(list(list_fd1.items())[0: K]) 
list_output_fd2 = dict(list(list_fd2.items())[0: K])

print("K highest frequency words are : " + str(list_output_fd1)) 
print("K highest frequency words are : " + str(list_output_fd2)) 

mplp.bar(list_output_fd1.keys(),list_output_fd1.values())
mplp.show()
mplp.bar(list_output_fd2.keys(),list_output_fd2.values())
mplp.show()

word_cloud_instance = WordCloud(width = 800, height = 800, background_color ='black',  
                      min_font_size = 8).generate(T2) 
                     
mplp.figure(figsize = (8, 8), facecolor = None) 
mplp.imshow(word_cloud_instance) 
mplp.axis("off") 
mplp.tight_layout(pad = 0) 
mplp.show() 


nltk.download('stopwords')

from nltk.corpus import stopwords
all_stopwords = stopwords.words('english')

word_cloud_instance = WordCloud(width = 800, height = 800, background_color ='black',  
                      stopwords = all_stopwords,min_font_size = 8).generate(T1) 
                     
mplp.figure(figsize = (8, 8), facecolor = None) 
mplp.imshow(word_cloud_instance) 
mplp.axis("off") 
mplp.tight_layout(pad = 0) 
mplp.show() 

word_cloud_instance = WordCloud(width = 800, height = 800, background_color ='black',  
                      stopwords = all_stopwords,min_font_size = 8).generate(T2) 
                     
mplp.figure(figsize = (8, 8), facecolor = None) 
mplp.imshow(word_cloud_instance) 
mplp.axis("off") 
mplp.tight_layout(pad = 0) 
mplp.show() 

from collections import defaultdict

words = {}

def word_counter(text):
   

    for word in text.split():
          
        if(len(word) not in words):
        	words[len(word)]=1
        else:
        	words[len(word)]+=1

word_counter(T1)

list_count_t1 = sorted(words.items())
x1,y1=zip(*list_count_t1)
mplp.plot(x1,y1)
mplp.xticks(range(0,20))
mplp.rcParams["figure.figsize"] = (10,5)
mplp.xlabel("Wordlength")
mplp.ylabel("Frequency")
mplp.show()

word_counter(T2)

list_count_t2 = sorted(words.items())
x2,y2=zip(*list_count_t2)
mplp.plot(x2,y2)
mplp.xticks(range(0,20))
mplp.rcParams["figure.figsize"] = (10,5)
mplp.xlabel("Wordlength")
mplp.ylabel("Frequency")
mplp.show()

mplp.plot(x1, y1, label = "T1")
mplp.plot(x2, y2, label = "T2")
mplp.xlabel('Wordlength')
mplp.ylabel('Frequency')
mplp.legend()
mplp.rcParams["figure.figsize"] = (10,5)
mplp.xticks(range(0,20))
mplp.show()



