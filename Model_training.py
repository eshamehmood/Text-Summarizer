#!/usr/bin/env python
# coding: utf-8

import argparse
import random as rn
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
from tqdm.auto import tqdm, trange
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1)

parser = argparse.ArgumentParser(description='run Model Training')

parser.add_argument('-NE',dest='dataset',default=False,help="Path to dataset Either dataset_with_NE or dataset_without_NE",type=bool)
parser.add_argument('-encoder_epoch',default=10 ,dest='e_epoch',help="Number of epochs for encoder",type=int)
parser.add_argument('-decoder_epoch',default=5,dest='d_epoch',help="Number of epochs for decoder",type=int)

args = parser.parse_args()

np.random.seed(42)
tf.random.set_seed(32)
rn.seed(12)

if args.dataset:
    post_pre=pd.read_csv('Dataset/dataset_with_NE.csv')
else:
    post_pre=pd.read_csv('Dataset/dataset_without_NE.csv')

max_text_len = 330
max_summary_len = 41

from sklearn.model_selection import train_test_split
x_train,x_validation,y_train,y_validation=train_test_split(np.array(post_pre['text']),np.array(post_pre['summary']),random_state=33 ,test_size=0.1)

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

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

optimizer = tf.keras.optimizers.Adam()

def nll_loss(p_vocab,target):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss = -p_vocab
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask  
    return loss

def coverage_loss(attention_weights,coverage_vector,target):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    coverage_vector = tf.expand_dims(coverage_vector,axis=2)
    ct_min = tf.reduce_min(tf.concat([attention_weights,coverage_vector],axis=2),axis=2)
    cov_loss = tf.reduce_sum(ct_min,axis=1)
    mask = tf.cast(mask, dtype=cov_loss.dtype)
    cov_loss *= mask
    return cov_loss

@tf.function
def training_step(encoder_input, decoder_target):
    """Function which performs one training step (batch)"""
    loss = tf.zeros(batch_size)
    lambda_cov = 1
    with tf.GradientTape() as tape:
        encoder_init_states = [tf.zeros((batch_size, encoder.hidden_units)) for i in range(2)]
        encoder_output, encoder_states = encoder(encoder_input,encoder_init_states)
        decoder_state = encoder_states[0] # !!!interpolate between forward and backward instead!!!
        coverage_vector = tf.zeros((64,encoder_input.shape[1]))
        for t in range(decoder_target.shape[1]-1):
            decoder_input_t = decoder_target[:,t]
            decoder_target_t = decoder_target[:,t+1]
            context_vector, attention_weights, coverage_vector = attention([decoder_state, encoder_output,coverage_vector])
            p_vocab,decoder_state = decoder(tf.expand_dims(decoder_input_t,1),decoder_state,encoder_output,context_vector)
            p_vocab_list = []
            for i in range(decoder_target_t.shape[0]):
                p_vocab_list.append(p_vocab[i,decoder_target_t[i]])
            p_vocab_target = tf.stack(p_vocab_list)
            loss += nll_loss(p_vocab_target,decoder_target_t) + lambda_cov*coverage_loss(attention_weights,coverage_vector,decoder_target_t)

        seq_len_mask = tf.cast(tf.math.logical_not(tf.math.equal(decoder_target, 0)),tf.float32)
        batch_seq_len = tf.reduce_sum(seq_len_mask,axis=1)

        batch_loss = tf.reduce_mean(loss/batch_seq_len)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(batch_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss

@tf.function
def validation_step(encoder_input, decoder_target):
    loss = tf.zeros(batch_size)
    lambda_cov = 1

    encoder_init_states = [tf.zeros((batch_size, encoder.hidden_units)) for i in range(2)]
    encoder_output, encoder_states = encoder(encoder_input,encoder_init_states)
    decoder_state = encoder_states[0] 
    coverage_vector = tf.zeros((64,encoder_input.shape[1]))

    for t in range(decoder_target.shape[1]-1):
        decoder_input_t = decoder_target[:,t]
        decoder_target_t = decoder_target[:,t+1]
        context_vector, attention_weights, coverage_vector = attention([decoder_state, encoder_output,coverage_vector])
        p_vocab,decoder_state = decoder(tf.expand_dims(decoder_input_t,1),decoder_state,encoder_output,context_vector)
        p_vocab_list = []
            
        for i in range(decoder_target_t.shape[0]):
            p_vocab_list.append(p_vocab[i,decoder_target_t[i]])
            
        p_vocab_target = tf.stack(p_vocab_list)
        loss += nll_loss(p_vocab_target,decoder_target_t) + lambda_cov*coverage_loss(attention_weights,coverage_vector,decoder_target_t)

        
    seq_len_mask = tf.cast(tf.math.logical_not(tf.math.equal(decoder_target, 0)),tf.float32)
    batch_seq_len = tf.reduce_sum(seq_len_mask,axis=1)

    val_batch_loss = tf.reduce_mean(loss/batch_seq_len)

    return val_batch_loss

from datetime import datetime
import time
import math

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


epochs = args.e_epoch

epoch_loss = tf.keras.metrics.Mean()
epoch_val_loss = tf.keras.metrics.Mean()
with tqdm(total=epochs) as epoch_progress:
    for epoch in range(epochs):
        epoch_loss.reset_states()
        epoch_val_loss.reset_states()

        with tqdm(total=len(body_seqs) // batch_size) as batch_progress:
            for batch, (encoder_input, decoder_target) in enumerate(train_dataset):
                batch_loss = training_step(encoder_input, decoder_target)
                epoch_loss(batch_loss)
                with train_summary_writer.as_default():
                    
                    tf.summary.scalar('train loss', batch_loss, step=epoch)    

                if (batch % 10) == 0:
                    batch_progress.set_description(f'Epoch {epoch + 1}')
                    batch_progress.set_postfix(Batch=batch, Loss=batch_loss.numpy())
                batch_progress.update()

        with tqdm(total=len(body_seqs_val) // batch_size) as batch_progress:
            for batch, (encoder_input_val, decoder_target_val) in enumerate(val_dataset):
                val_batch_loss = validation_step(encoder_input_val, decoder_target_val)
                epoch_val_loss(val_batch_loss)

                with test_summary_writer.as_default():
                    
                    tf.summary.scalar('test loss', val_batch_loss, step=epoch)
                
                if (batch % 10) == 0:
                    batch_progress.set_description(f'Epoch {epoch + 1}')
                    batch_progress.set_postfix(Batch=batch, Loss=batch_loss.numpy())
                batch_progress.update()

        epoch_progress.set_description(f'Epoch {epoch + 1}')
        epoch_progress.set_postfix(Loss=epoch_loss.result().numpy())
        print('train loss_loss = ',epoch_loss.result().numpy())

        print('val_loss = ',epoch_val_loss.result().numpy())

        epoch_progress.update() 

d_epochs = args.d_epoch

epoch_loss = tf.keras.metrics.Mean()
epoch_val_loss = tf.keras.metrics.Mean()
with tqdm(total=d_epochs) as epoch_progress:
    for epoch in range(epoch,epoch+d_epochs):
        epoch_loss.reset_states()
        epoch_val_loss.reset_states()

        with tqdm(total=len(body_seqs) // batch_size) as batch_progress:
            for batch, (encoder_input, decoder_target) in enumerate(train_dataset):
                batch_loss = training_step(encoder_input, decoder_target)
                epoch_loss(batch_loss)
                with train_summary_writer.as_default():
                    
                    tf.summary.scalar('train loss', batch_loss, step=epoch)

                if (batch % 10) == 0:
                    batch_progress.set_description(f'Epoch {epoch + 1}')
                    batch_progress.set_postfix(Batch=batch, Loss=batch_loss.numpy())
                batch_progress.update()

        with tqdm(total=len(body_seqs_val) // batch_size) as batch_progress:
            for batch, (encoder_input_val, decoder_target_val) in enumerate(val_dataset):
                val_batch_loss = validation_step(encoder_input_val, decoder_target_val)
                epoch_val_loss(val_batch_loss)

                with test_summary_writer.as_default():
                    
                    tf.summary.scalar('test loss', val_batch_loss, step=epoch)
                
                if (batch % 10) == 0:
                    batch_progress.set_description(f'Epoch {epoch + 1}')
                    batch_progress.set_postfix(Batch=batch, Loss=batch_loss.numpy())
                batch_progress.update()

        epoch_progress.set_description(f'Epoch {epoch + 1}')
        epoch_progress.set_postfix(Loss=epoch_loss.result().numpy())
        print('train loss_loss = ',epoch_loss.result().numpy())

        print('val_loss = ',epoch_val_loss.result().numpy())

        epoch_progress.update() 

if args.dataset:
    ne_name = 'NE'
else:
    ne_name = 'WNE'

encoder_save_name = "Models/encoder_"+str(epochs)+"epochs_"+ne_name+".h5"
decoder_save_name = "Models/decoder_"+str(d_epochs)+"epochs_"+ne_name+".h5"

encoder.save_weights(encoder_save_name)
decoder.save_weights(decoder_save_name)

print("Model Training Complete")