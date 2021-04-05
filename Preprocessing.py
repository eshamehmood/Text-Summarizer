#!/usr/bin/env python
# coding: utf-8

from os import listdir
import warnings
import string
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import seaborn as sns
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1)

def load_articles(file_name):
    file = open(file_name, encoding='utf-8')
    text = file.read()
    file.close()
    return text

def spliting_story(doc):
    index = doc.find('@highlight')
    story, highlights = doc[:index], doc[index:].split('@highlight')
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights


stories = list()
directory = 'Dataset/cnn/stories'
for name in tqdm(listdir(directory)):
    filename = directory + '/' + name
    doc = load_articles(filename)
    story, highlights = spliting_story(doc)
    stories.append({'story':story, 'highlights':highlights})

cnn_df = pd.DataFrame.from_dict(stories)
cnn_df.columns = ['article','summary']

cnn_df.to_csv('Dataset/Dataset.csv',index=False)

CNN=pd.read_csv('Dataset/Dataset.csv')

def decontraction(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

article_text=[]
for i in CNN.article.values:
    temp_text=re.sub(r'\n',' ', i)
    temp_text=re.sub(r'>',' ', temp_text)
    temp_text=re.sub(r'<',' ', temp_text)
    temp_text=re.sub(r'(CNN)',' ', temp_text)
    temp_text=re.sub(r'LRB',' ', temp_text)
    temp_text=re.sub(r'RRB',' ', temp_text)
    temp_text = re.sub(r'[" "]+', " ", temp_text)
    temp_text=re.sub(r'-- ',' ', temp_text)
    temp_text=re.sub(r"([?!¿])", r" \1 ", temp_text)
    temp_text=re.sub(r'-',' ', temp_text)
    temp_text=temp_text.replace('/',' ')
    temp_text=re.sub(r'\s+', ' ', temp_text)
    temp_text=decontraction(temp_text)
    temp_text = re.sub('[^A-Za-z0-9.,]+', ' ', temp_text)
    temp_text = temp_text.lower()
    article_text.append(temp_text)

data_article=pd.DataFrame(article_text,columns=['Article'])

summary_text=[]
for i in CNN.summary.values:
    temp_text=re.sub(r'\n',' ', i)
    temp_text=re.sub(r'>',' ', temp_text)
    temp_text=re.sub(r'<',' ', temp_text)
    temp_text=re.sub(r'-',' ', temp_text)
    temp_text=re.sub(r'(CNN)',' ', temp_text)
    temp_text=re.sub(r'LRB',' ', temp_text)
    temp_text=re.sub(r'RRB',' ', temp_text)
    temp_text = re.sub(r'[" "]+', " ", temp_text)
    temp_text=re.sub(r'-- ',' ', temp_text)
    temp_text=temp_text.replace('/',' ')
    temp_text=re.sub(r'\s+', ' ', temp_text)
    temp_text=decontraction(temp_text)
    temp_text = re.sub('[^A-Za-z0-9.]+', ' ', temp_text)
    temp_text = temp_text.lower()
    summary_text.append(temp_text)

summary_text=np.array(summary_text)
summary_text=summary_text.reshape(-1,1)

data_summ=pd.DataFrame(summary_text,columns=['Summary'])

data_cleaned=data_article.join(data_summ)

data_cleaned.to_csv('Dataset/cleaned_dataset.csv',index=False)

max_text_len = 330
max_summary_len = 40

cleaned_text = np.array(data_cleaned['Article'])
cleaned_summary = np.array(data_cleaned['Summary'])

short_text = []
short_summary = []

for i in tqdm(range(len(cleaned_text))):
    if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
        doc1 = nlp(cleaned_text[i])    
        c=(" ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc1]))
        c=c.lower()
        short_text.append(c)

        doc2 = nlp(cleaned_summary[i])
        k=(" ".join([t.text if not t.ent_type_ else t.ent_type_ for t in doc2]))
        k=k.lower()
        short_summary.append(k)                
        
post_pre=pd.DataFrame({'text':short_text,'summary':short_summary})

post_pre.to_csv('Dataset/dataset_with_NE.csv')

short_text = []
short_summary = []

for i in tqdm(range(len(cleaned_text))):
    if(len(cleaned_summary[i].split())<=max_summary_len and len(cleaned_text[i].split())<=max_text_len):
        doc1 = nlp(cleaned_text[i])
        c=(" ".join([t.text for t in doc1]))
        short_text.append(c)

        doc2 = nlp(cleaned_summary[i])
        k=(" ".join([t.text for t in doc2]))
        short_summary.append(k)
        
post_pre=pd.DataFrame({'text':short_text,'summary':short_summary})

post_pre.to_csv('Dataset/dataset_without_NE.csv')

print('Preprocessing Complete')