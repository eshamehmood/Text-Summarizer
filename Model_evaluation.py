#!/usr/bin/env python
# coding: utf-8

import random as rn
import argparse
from tqdm import tqdm
import re
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K 
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed,Add,AdditiveAttention
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from sumeval.metrics.rouge import RougeCalculator

warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1)
rn.seed(12)
np.random.seed(42)
tf.random.set_seed(32)

parser = argparse.ArgumentParser(description='run Model Evaluation')

parser.add_argument('-NE',dest='dataset',default=False,help="Path to dataset Either dataset_with_NE or dataset_without_NE",type=bool)
parser.add_argument('-encoder_name',default=None ,dest='en',help="encoder.h5 file name in models folder ")
parser.add_argument('-decoder_name',default=None,dest='de',help="decoder.h5 file name in models folder")

args = parser.parse_args()

if args.dataset:
    post_pre=pd.read_csv('Dataset/dataset_with_NE.csv')
else:
    post_pre=pd.read_csv('Dataset/dataset_without_NE.csv')

max_text_len = 330
max_summary_len = 41

x_train,x_validation,y_train,y_validation=train_test_split(np.array(post_pre['text']),np.array(post_pre['summary']),random_state=33 ,test_size=0.1)

x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_train))

thresh=2
rare_word=[]
for key,value in x_tokenizer.word_counts.items():
    if(value<thresh):
        rare_word.append(key)

tokenrare=[]
for i in range(len(rare_word)):
    tokenrare.append('ukn')

dictionary = dict(zip(rare_word,tokenrare)) 

x_trunk=[]
for i in x_train:
    for word in i.split():
        if word.lower() in dictionary:
            i = i.replace(word, dictionary[word.lower()])
    x_trunk.append(i)

x_tokenizer = Tokenizer(oov_token='ukn') 
x_tokenizer.fit_on_texts(list(x_trunk))

x_tr_seq    =   x_tokenizer.texts_to_sequences(x_trunk) 
x_val_seq   =   x_tokenizer.texts_to_sequences(x_validation)

x_tr    =   pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val   =   pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

x_vocab   =  len(x_tokenizer.word_index) + 1

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
y_tokenizer = Tokenizer() 
y_tokenizer.fit_on_texts(list(y_train))

thresh=2
rare_word=[]
for key,value in y_tokenizer.word_counts.items():
    if(value<thresh):
        rare_word.append(key)

tokenrare=[]
for i in range(len(rare_word)):
    tokenrare.append('ukn')

dictionary = dict(zip(rare_word,tokenrare)) 

y_trunk=[]
for i in y_train:
    for word in i.split():
        if word.lower() in dictionary:
            i = i.replace(word, dictionary[word.lower()])
    y_trunk.append(i)

y_tokenizer = Tokenizer(oov_token='ukn') 
y_tokenizer.fit_on_texts(list(y_trunk))

y_tr_seq    =   y_tokenizer.texts_to_sequences(y_trunk) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_validation) 

y_tr    =   pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

y_vocab  =   len(y_tokenizer.word_index) +1

embeddings_dictionary = dict()
glove_file = open("Files/glove.42B.300d.txt", encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix_x = np.zeros((x_vocab+1 , 300))
for word, index in x_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix_x[index] = embedding_vector
embedding_matrix_x.shape

embedding_matrix_y = np.zeros((y_vocab+1, 300))

for word, index in y_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix_y[index] = embedding_vector
embedding_matrix_y.shape

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim,embedding_matrix_x, hidden_units):
        super().__init__()
        
        self.hidden_units = hidden_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,weights=[embedding_matrix_x])
        self.bi_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',dropout=0.08,recurrent_dropout=0.05))
        
    def call(self, encoder_input,encoder_states):
        encoder_emb = self.embedding(encoder_input)
        
        encoder_output, state_fwd, state_back = self.bi_gru(encoder_emb,initial_state=encoder_states)
        encoder_states = [state_fwd,state_back]

        return encoder_output, encoder_states

class additiveAttention(tf.keras.layers.AdditiveAttention):
    def __init__(self, hidden_units,is_coverage=False):
        super().__init__()
        self.Wh = tf.keras.layers.Dense(hidden_units)
        self.Ws = tf.keras.layers.Dense(hidden_units)
        self.wc = tf.keras.layers.Dense(1)
        self.V = tf.keras.layers.Dense(1)
        self.coverage = is_coverage
        if self.coverage is False:
            self.wc.trainable = False
        
    def call(self,keys):
        value=keys[0]
        query=keys[1]
        ct=keys[2]
        value = tf.expand_dims(value, 1)
        ct = tf.expand_dims(ct, 1)
        score = self.V(tf.nn.tanh(
                        self.Wh(query) +
                        self.Ws(value) +  
            
                        self.wc(ct) 
                        )) 
        attention_weights = tf.nn.softmax(score, axis=1) 
        ct = tf.squeeze(ct,1)
        if self.coverage is True:
            ct+=tf.squeeze(attention_weights) 
        context_vector = attention_weights * query 
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights, ct

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim,embedding_matrix_y,hidden_units):
        super().__init__()
        
        self.hidden_units = hidden_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,weights=[embedding_matrix_y])
        self.gru = tf.keras.layers.GRU(
            hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
        )
        self.W1 = tf.keras.layers.Dense(hidden_units)
        self.W2 = tf.keras.layers.Dense(vocab_size)
        
        # self.wh = tf.keras.layers.Dense(1)
        # self.ws = tf.keras.layers.Dense(1)
        # self.wx = tf.keras.layers.Dense(1)

    def call(self, decoder_input, decoder_state, encoder_output,context_vector):
        decoder_emb = self.embedding(decoder_input)
        decoder_output , decoder_state = self.gru(decoder_emb,initial_state=decoder_state)
        concat_vector = tf.concat([context_vector,decoder_state], axis=-1)
        concat_vector = tf.reshape(concat_vector, (-1, concat_vector.shape[1]))
        p_vocab = tf.nn.log_softmax(self.W2(self.W1(concat_vector)))

        # p_gen = tf.nn.sigmoid(self.wh(context_vector)+self.ws(decoder_state)+self.wx(decoder_input))
          
        return p_vocab, decoder_state

input_vocab_size = x_vocab+1
output_vocab_size = y_vocab +1

def data_generator(X,y,BATCH_SIZE,shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(len(X)).batch(BATCH_SIZE,drop_remainder=True)
    else:
        dataset = dataset.batch(BATCH_SIZE,drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

body_seqs=x_tr
target_seqs=y_tr

body_seqs_val=x_val
target_seqs_val=y_val

train_dataset = data_generator(body_seqs,target_seqs,BATCH_SIZE=64,
                       shuffle=True)
val_dataset = data_generator(body_seqs_val,target_seqs_val,BATCH_SIZE=64,
                       shuffle=False)

embedding_dim = 300
hidden_units = 128
batch_size=64

encoder = Encoder(input_vocab_size, embedding_dim,embedding_matrix_x, hidden_units) 
attention = additiveAttention(hidden_units,is_coverage=True)
decoder = Decoder(output_vocab_size, embedding_dim,embedding_matrix_y,hidden_units)

encoder_input, decoder_target = next(iter(train_dataset))
encoder_init_states = [tf.zeros((batch_size, encoder.hidden_units)) for i in range(2)]
encoder_output, encoder_states = encoder(encoder_input,encoder_init_states)
decoder_state = encoder_states[0] 
coverage_vector = tf.zeros((64,encoder_input.shape[1]))
decoder_input_t = decoder_target[:,0]
context_vector, attention_weights, coverage_vector = attention([decoder_state,encoder_output,coverage_vector])
p_vocab,decoder_state = decoder(tf.expand_dims(decoder_input_t,1),decoder_state,encoder_output,context_vector)

if args.en == None:
    if args.dataset:
        args.en = "encoder_NE.h5"
    else:
        args.en = "encoder_WNE.h5"

if args.de == None:
    if args.dataset:
        args.de = "decoder_NE.h5"
    else:
        args.de = "decoder_WNE.h5"

encoder.load_weights('Models/'+args.en)
decoder.load_weights('Models/'+args.de)

def decode_sequence(encoder_input):
    """Function which returns a summary by always picking the highest probability option conditioned on the previous word"""
    encoder_init_states = [tf.zeros((1, encoder.hidden_units)) for i in range(2)]
    encoder_output, encoder_states = encoder(encoder_input,encoder_init_states)
    decoder_state = encoder_states[0]

    decoder_input_t =  tf.ones(1)*target_word_index['start'] # initialize with start token
    summary = [target_word_index['start']]
    coverage_vector = tf.zeros((1,encoder_input.shape[1]))
    while decoder_input_t[0].numpy()!=target_word_index['end'] and len(summary)<max_summary_len: # as long as decoder input is different from end token continue
        context_vector, attention_weights, coverage_vector = attention([decoder_state, encoder_output,coverage_vector])
        p_vocab, decoder_state = decoder(tf.expand_dims(decoder_input_t,1),decoder_state,encoder_output,context_vector)
        decoder_input_t = tf.argmax(p_vocab,axis=1) 
        decoder_word_idx = int(decoder_input_t[0].numpy())
        summary.append(decoder_word_idx)
    return summary

reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

def sequence_to_text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString

def sequence_to_summary(input_seq,ukn_token):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['start']) and i!=target_word_index['end']):
            if i==target_word_index['ukn']:

                newString=newString+ukn_token+' '
            else:     
                newString=newString+reverse_target_word_index[i]+' '
    return newString

def search(list, platform):
    for i in range(len(list)):
        if list[i] == platform:
            return True
    return False

rouge = RougeCalculator(stopwords=True, lang="en")

Rouge_1 = []
Rouge_2 = []
Rouge_L = []

for i in tqdm(range(0,len(x_val)+1)):
    encoder_input_sum = tf.expand_dims(x_val[i],0)
    summary = decode_sequence(encoder_input_sum)
    k= sequence_to_text(x_val[i])
    k=re.sub('[^a-z]+', ' ', k)
    result = text_to_word_sequence(k)
    if search(result, 'ukn'):
        idx=result.index('ukn')
        input_org = re.sub('[^a-z]+',' ', x_validation[i])
        input_org = text_to_word_sequence(input_org)
        ukn_token = input_org[idx]
    else:
        ukn_token='ukn'
    
    r_1 = rouge.rouge_n(
            summary=sequence_to_summary(y_val[i], ukn_token),
            references=sequence_to_summary(summary, ukn_token),
            n=1)
    Rouge_1.append(r_1)
   
    r_2 = rouge.rouge_n(
         summary=sequence_to_summary(y_val[i], ukn_token),
            references=sequence_to_summary(summary, ukn_token),
            n=2)
    Rouge_2.append(r_2)

   
    r_L = rouge.rouge_L(
           summary=sequence_to_summary(y_val[i], ukn_token),
            references=sequence_to_summary(summary, ukn_token))
    Rouge_L.append(r_L)

print("ROUGE-1: {}, ROUGE-2: {}, ROUGE-L: {}".format(np.round(sum(Rouge_1)/len(Rouge_1),3),
                                                     np.round(sum(Rouge_2)/len(Rouge_2),3),
                                                     np.round(sum(Rouge_L)/len(Rouge_L),3)
))