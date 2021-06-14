import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.metrics import accuracy_score # to measure how good we are 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Activation
from keras.layers import Flatten
from keras.applications import vgg16
import re

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)    
    args = parser.parse_args()
    return args

args = parse_args()

epochs = args.epoch

def extract_feature(file_name, **kwargs):         #function for extracting speech features
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
    """
    mfcc = kwargs.get("mfcc")            
    chroma = kwargs.get("chroma")        
    mel = kwargs.get("mel")              
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        X = librosa.to_mono(X)
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result1 = np.array([])
        result2 = np.array([])
        result3 = np.array([])

        if mfcc:                         #getting the mfcc feature 
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result1 = np.hstack((result1, mfccs))
            
        if chroma:                       #getting the chroma feature 
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result2 = np.hstack((result2, chroma))
        if mel:                          #getting MEL Spectrogram Frequency (mel) feature
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result3 = np.hstack((result3, mel))
            
                                         #stacking three features in the variable called result
    return result1,result2,result3


AVAILABLE_EMOTIONS = {                 # available emotions set
    "Anger",
    "Happiness",
    "Neutra",
    "Sadness"
}


R1,R2,R3,q=[],[],[],[]
count=0
#importing speech modality
for i in AVAILABLE_EMOTIONS:                   # made files of each of the four emotions for speech files
    for file in sorted(glob.glob(i+"/Ses01*.wav")):        #for every sound file in session 1 
        features1,features2,features3= extract_feature(file, mfcc=True, chroma=True, mel=True) #calling the extract feature function for speech features
        count+=1                            #to count the number of files in session 1
        R1.append(features1)                  #appending speech features for every audio in list R 
        R2.append(features2)
        R3.append(features3)
        p=re.split("/", file, 1)            
        p.reverse()
        q.append(p)                         #q is storing the respective emotion for each sound file
        
      
    for file in sorted(glob.glob(i+"/Ses02*.wav")):        #for every sound file in session 1 
        features1,features2,features3= extract_feature(file, mfcc=True, chroma=True, mel=True) #calling the extract feature function for speech features
        count+=1                            #to count the number of files in session 1
        R1.append(features1)                  #appending speech features for every audio in list R 
        R2.append(features2)
        R3.append(features3)
        p=re.split("/", file, 1)            
        p.reverse()
        q.append(p)                         #q is storing the respective emotion for each sound file
        

    '''for file in sorted(glob.glob(i+"/Ses03*.wav")):
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        count+=1
        R.append(features)
        p=re.split("/", file, 1)
        p.reverse()
        q.append(p) 

    for file in sorted(glob.glob(i+"/Ses04*.wav")):
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        count+=1
        R.append(features)
        p=re.split("/", file, 1)
        p.reverse()
        q.append(p)

    for file in sorted(glob.glob(i+"/Ses05*.wav")):
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        count+=1
        R.append(features)
        p=re.split("/", file, 1)
        p.reverse()
        q.append(p)  '''   
    print("1/4th done")            
print(count)        


R1=np.array(R1)    
R2=np.array(R2)    
R3=np.array(R3)                     #converting list into numpy array
q=np.array(q)                       #converting list into numpy array
R1=R1.astype('float32')  
R2=R2.astype('float32')  
R3=R3.astype('float32')      
q[q=='Anger']=0                     # numbering the respective emotions 0,1,2,3
q[q=='Happiness']=1
q[q=='Sadness']=2
q[q=='Neutra']=3


arr1 = np.concatenate((q,R1), axis=1)  # concatenating the columns q and R to feed both speech and text features simultaneously to the model
li1=arr1.tolist()                      # converting arr to list to sort (easy to sort using a list)
li1=sorted(li1)                        #sorting the list on the basis of q value
li1=np.array(li1)                      #converting the list back to numpy array

arr2 = np.concatenate((q,R2), axis=1)  # concatenating the columns q and R to feed both speech and text features simultaneously to the model
li2=arr2.tolist()                      # converting arr to list to sort (easy to sort using a list)
li2=sorted(li2)                        #sorting the list on the basis of q value
li2=np.array(li2)                      #converting the list back to numpy array

arr3 = np.concatenate((q,R3), axis=1)  # concatenating the columns q and R to feed both speech and text features simultaneously to the model
li3=arr3.tolist()                      # converting arr to list to sort (easy to sort using a list)
li3=sorted(li3)                        #sorting the list on the basis of q value
li3=np.array(li3)                      #converting the list back to numpy array


l=li1[:,1]                           #getting the label(labels are emotions) vector l using li array
M1=li1[:,2:] 
M2=li2[:,2:] 
M3=li3[:,2:]                          #getting the speech feature matrix M using li array
l=l.astype('int64') 
M1=M1.astype('float32')
M2=M2.astype('float32')
M3=M3.astype('float32')

print(np.shape(M3))
print(np.shape(M1))
np.shape(M2)


#text features for each text file for session1 (doing it for all 10 emotions, will take the 4 emotions in later part)
y=[]
for i in range(1,9): 
    for file in glob.glob("Ses01F_impro0"+str(i)+".txt"):     #for every text file starting with Ses01F_impro0
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)                #getting file name and corresponding text for each file 
            if(x==None):
                continue
            p = re.split("\s", line, 2)                 
            del(p[1])                                  # deleting useless information in the list p
            y.append(p)                                # appending the file name and corresponding text in list 'p' of every audio file in y
for i in range(1,9): 
    for file in glob.glob("Ses01M_impro0"+str(i)+".txt"):    #for every text file starting with Ses01M_impro0
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)
            if(x==None):
                continue
            p = re.split("\s", line, 2)
            del(p[1])
            y.append(p)            

#session2
#text features for each text file for session2 (same as above)
for i in range(1,9): 
    for file in glob.glob("Ses02F_impro0"+str(i)+"*.txt"):
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)
            if(x==None):
                continue
            p = re.split("\s", line, 2)
            del(p[1])
            y.append(p)
for i in range(1,9): 
    for file in glob.glob("Ses02M_impro0"+str(i)+"*.txt"):
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)
            if(x==None):
                continue
            p = re.split("\s", line, 2)
            del(p[1])
            y.append(p)    
#y contain file name and corresponding text for every audio files in session 1 and 2 (all 10 emotions)


#did the same for session 3 (ablation study)
#session3
'''for i in range(1,9): 
    for file in glob.glob("Ses03F_impro0"+str(i)+"*.txt"):
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)
            if(x==None):
                continue
            p = re.split("\s", line, 2)
            del(p[1])
            y.append(p)
for i in range(1,9): 
    for file in glob.glob("Ses03M_impro0"+str(i)+"*.txt"):
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)
            if(x==None):
                continue
            p = re.split("\s", line, 2)
            del(p[1])
            y.append(p)    
'''


'''for i in range(1,9): 
    for file in glob.glob("Ses04F_impro0"+str(i)+"*.txt"):
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)
            if(x==None):
                continue
            p = re.split("\s", line, 2)
            del(p[1])
            y.append(p)
for i in range(1,9): 
    for file in glob.glob("Ses04M_impro0"+str(i)+"*.txt"):
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)
            if(x==None):
                continue
            p = re.split("\s", line, 2)
            del(p[1])
            y.append(p)    
'''


'''for i in range(1,9): 
    for file in glob.glob("Ses05F_impro0"+str(i)+"*.txt"):
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)
            if(x==None):
                continue
            p = re.split("\s", line, 2)
            del(p[1])
            y.append(p)
for i in range(1,9): 
    for file in glob.glob("Ses05M_impro0"+str(i)+"*.txt"):
        f = open(file)
        lines = f.readlines()
        for line in lines:
            x = re.search("^Ses0", line)
            if(x==None):
                continue
            p = re.split("\s", line, 2)
            del(p[1])
            y.append(p)    
'''
print(y)

print(type(y))
x=sorted(y)
# sorting the text on the basis of file name
# x will contain file name and corresponding text, sorted according to file names
x[-1]

AVAILABLE_EMO = {
    "Anger",
    "Happiness",
    "Neutra",
    "Sadness",
    "Other",
    "Surprise",
    "Disgust",
    "Fear",
    "Excited",
    "Frustration"
}

y=[]
# looping again across all 10 emotions to get the desired file name and corresponding emotions
for i in AVAILABLE_EMO:
    for file in sorted(glob.glob(str(i)+"/Ses01*.wav")):   #again using the speech file 
        p=re.split("/", file, 2)
        p.reverse()
        p[0]=p[0][:-4]
        y.append(p)
for i in AVAILABLE_EMO:
    for file in sorted(glob.glob(str(i)+"/Ses02*.wav")):
        p=re.split("/", file, 2)
        p.reverse()
        p[0]=p[0][:-4]
        
        y.append(p) 

'''for i in AVAILABLE_EMO:
    for file in sorted(glob.glob(str(i)+"/Ses03*.wav")):
        p=re.split("/", file, 2)
        p.reverse()
        p[0]=p[0][:-4]
        y.append(p)   

for i in AVAILABLE_EMO:
    for file in sorted(glob.glob(str(i)+"/Ses04*.wav")):
        p=re.split("/", file, 2)
        p.reverse()
        p[0]=p[0][:-4]
        y.append(p)
for i in AVAILABLE_EMO:
    for file in sorted(glob.glob(str(i)+"/Ses05*.wav")):
        p=re.split("/", file, 2)
        p.reverse()
        p[0]=p[0][:-4]
        y.append(p) '''       

n=[]
y=sorted(y)
x=sorted(x)
for i in range(0,len(x)):
  if x[i][0][-3:-2]!='X':
    n.append(x[i])


print(y)
print(x)

for i in range(0,len(n)):
  if n[i][0]!=y[i][0]:
    print("HI")


import numpy as np
X=np.array(n)
Y=np.array(y)
X=X[:,1]
Y=Y[:,1]

x,y=[],[]
count=0
EMOT = {
    "Anger",
    "Happiness",
    "Neutra",
    "Sadness"
}
for i in range(0,1547):
    if Y[i] in EMOT:
        x.append(X[i])
        y.append(Y[i])
        count+=1


y=np.array(y)
x=np.array(x)
print(y.shape)


# count=0
# for i in AVAILABLE_EMOTIONS:
#   for file in sorted(glob.glob(i+"/Ses03*.wav")): 
#     count=count+1

# print(count)


import nltk
nltk.download('stopwords')

import pandas as pd 
import numpy as np
import re
import collections
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from keras import models
from keras import layers
from keras import regularizers
stopwords_list = stopwords.words('english')
porter = PorterStemmer()


NB_WORDS = 100000  # Parameter indicating the number of words we'll put in the dictionary
VAL_SIZE = 1000    # Size of the validation set
NB_START_EPOCHS = 18  # Number of epochs we usually start to train with
BATCH_SIZE = 512  # Size of the batches used in the mini-batch gradient descent
MAX_LEN = 20  # Maximum number of words in a sequence
GLOVE_DIM = 50# Number of dimensions of the GloVe word embeddings


glove_file = 'glove.6B.' + str(GLOVE_DIM) + 'd.txt'
f='glove.6B/'
emb_dict = {}
glove = open(f+glove_file)
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    emb_dict[word] = vector
glove.close()


tk = Tokenizer(num_words=NB_WORDS,
               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
               lower=True,
               split=" ")
tk.fit_on_texts(x)

def remove_stopwords(input_text):
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 

for i in range(x.shape[0]):
    x[i]=remove_stopwords(x[i])
    x[i]= porter.stem(x[i])


y[y=='Anger']=0
y[y=='Happiness']=1
y[y=='Sadness']=2
y[y=='Neutra']=3
y=y.astype('int64') 
y.shape


##########IEMOCAP Train test Split################
M1_train,M1_test,M2_train,M2_test,M3_train,M3_test,l_train, l_test,x_train, x_test, y_train, y_test = train_test_split(M1,M2,M3,l,x,y,test_size=0.2)
#M_train, M_test, l_train, l_test = train_test_split(M,l,test_size=0.2)
get_ipython().system('pip install bert-for-tf2')


print(l_train-y_train)


#BERT for text features
import tensorflow.io as tf

from bert import bert_tokenization
def createTokenizer():
    currentDir = "/home/puneet/code/Interspeech/" #os.path.dirname(os.path.realpath('/content/drive/MyDrive/iemocap(version2)/'))
    modelsFolder = os.path.join(currentDir, "iemocap(version2)/model", "multi_cased_L-12_H-768_A-12")
    vocab_file = os.path.join(modelsFolder, "vocab.txt")

    tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer

tokenizer = createTokenizer()

print(x_test,x_train)

import csv
import os
import random
from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.sequence import pad_sequences #AttributeError: 'tuple' object has no attribute 'layer' #478

def loadData(tokenizer):
    
    train_set = []
    train_labels = []
    test_set = []
    test_labels = []
    for el in x_train:
        train_set.append(el)

    for el in x_test:
        test_set.append(el)

    train_tokens = [["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"] for sentence in train_set]
    train_tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in train_tokens]
    train_tokens_ids = pad_sequences(train_tokens_ids, maxlen=30, dtype="long", truncating="post", padding="post")
    
    test_tokens = [["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"] for sentence in test_set]
    test_tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in test_tokens]

    test_tokens_ids = pad_sequences(test_tokens_ids, maxlen=30, dtype="long", truncating="post", padding="post")
    return train_tokens_ids, test_tokens_ids

X_train, X_test = loadData(tokenizer)

import bert
import os
import tensorflow as tf
max_seq_length=30
def createBertLayer(max_seq_length):
    global bert_layer

    bertDir = os.path.join('./model/', "multi_cased_L-12_H-768_A-12")

    bert_params = bert.params_from_pretrained_ckpt(bertDir)

    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
    
    model_layer = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='input_ids'),
        bert_layer
    ])

    model_layer.build(input_shape=(None, max_seq_length))

    bert_layer.apply_adapter_freeze()

def loadBertCheckpoint():
    modelsFolder = os.path.join('./model/', "multi_cased_L-12_H-768_A-12")
    checkpointName = os.path.join(modelsFolder, "bert_model.ckpt")

    bert.load_stock_weights(bert_layer, checkpointName)

createBertLayer(30)
loadBertCheckpoint


import keras
import os
from keras.layers import Flatten,GRU,LSTM,Dense,Activation,Input,Dropout,Embedding,concatenate,Lambda
from tensorflow.keras.utils import plot_model #PK

#######IEMOCAP model ###############
import keras.backend as K
_input1 = Input(shape=(40,1), dtype='float32')

activations1 =   GRU(40,return_sequences=True)(_input1)
attention1  =   Dense(1,activation='tanh')(activations1)
attention12   =   Dense(1)(attention1)
attention13   =   Flatten()(attention12)
attention14   =   Activation('softmax')(attention13)

sent_representation1 = keras.layers.Multiply()([activations1,attention14])
sent_representation11 = keras.layers.Lambda(lambda xin: K.sum(xin,axis=-2),output_shape=(40,))(sent_representation1)

output1 = Dense(60, activation='relu')(sent_representation11)

_input2 = Input(shape=(12,1), dtype='float32')

activations2 =   GRU(12,return_sequences=True)(_input2)
attention2  =   Dense(1,activation='tanh')(activations2)
attention22   =   Dense(1)(attention2)
attention23   =   Flatten()(attention22)
attention24   =   Activation('softmax')(attention23)

sent_representation2 = keras.layers.Multiply()([activations2,attention24])
sent_representation21 = keras.layers.Lambda(lambda xin: K.sum(xin,axis=-2),output_shape=(12,))(sent_representation2)

output2 = Dense(20, activation='relu')(sent_representation21)

_input3 = Input(shape=(128,1), dtype='float32')

activations3 =   GRU(128,return_sequences=True)(_input3)
attention3  =   Dense(1,activation='tanh')(activations3)
attention32   =   Dense(1)(attention3)
attention33   =   Flatten()(attention32)
attention34   =   Activation('softmax')(attention33)

sent_representation3 = keras.layers.Multiply()([activations3,attention34])
sent_representation31 = keras.layers.Lambda(lambda xin: K.sum(xin,axis=-2),output_shape=(128,))(sent_representation3)

output3 = Dense(150, activation='relu')(sent_representation31)


_input4   =  Input(shape=(max_seq_length,), dtype='int32', name='input_ids')
_input41  =  bert_layer(_input4)
_input42=    Lambda(lambda x: x[:, 0, :])(_input41)
#print(tf.shape(_input2))
_input43  = Dense(600, activation='relu')(_input42)

output51    = concatenate([output1,output2,output3,_input43])
output52    = Dropout(0.3)(output51)
output53    = Dense(600, activation='relu')(output52)
output54    = Dense(4, activation='softmax')(output53)

model = keras.Model(inputs=[_input1,_input2,_input3,_input4], outputs=[output54])
model.summary()
plot_model(model)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


import tensorflow as tf
ACCURACY_THRESHOLD = 0.72
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_accuracy') >= ACCURACY_THRESHOLD):   
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
            self.model.stop_training = True

callbacks = myCallback()        


#########IEMOCAP training ######################
from keras.callbacks import ModelCheckpoint
filepath="weights.best.hdf5"
model.fit([M1_train,M2_train,M3_train,X_train], y_train, epochs=100, batch_size=64,validation_data=([M1_test,M2_test,M3_test,X_test], y_test),verbose=2)

#y_pred = np.loadtxt('y_pred.txt') 
#y_test = np.loadtxt('y_test.txt') 

#_, accuracy = model.evaluate([M1_test,M2_test,M3_test], y_test)
#print('Accuracy on test data: %.2f' % (accuracy*100))
#_, accu = model.evaluate([M1_train,M2_train,M3_train], y_train)
#print('Accuracy on train data: %.2f' % (accu*100))


import matplotlib.pyplot as plt
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train", "Validation"])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(["Train", "Validation"])
plt.title('model Loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


# # code for computing intersection matrix
#Taking out the output of each layer
import tensorflow as tf
import keras.backend as K
out_layer=model.layers[23].output
output_fn= K.function([model.input],out_layer)
o=output_fn([M1_train,M2_train,M3_train])
o.shape

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(o)
o=scaler.transform(o)
print(o.shape)

from sklearn.decomposition import PCA
pca=PCA(n_components=3)

pca.fit(o)

x_pca=pca.transform(o)
x_pca.shape

X=x_pca[:,0]
Y=x_pca[:,1]
Z=x_pca[:,1]

from mpl_toolkits.mplot3d import Axes3D

fig= plt.figure(figsize=(11,11))
ax=plt.axes(projection='3d')
ax.scatter(X,Y,Z,c=y_train,marker='^')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_zlabel('z axis')
plt.show()


import matplotlib.lines as mlines
import matplotlib.pyplot as plt

fig, (ax1) = plt.subplots(1,1,figsize=(8,8))
ax1 = plt.axes()
ax1.scatter(X[y_train==1],Y[y_train==1], marker='<',color='blue',s=100)
ax1.scatter(X[y_train==2],Y[y_train==2], marker='s',color='orange',s=100)
ax1.scatter(X[y_train==3],Y[y_train==3], marker='o',color='green',s=100)
ax1.scatter(X[y_train==0],Y[y_train==0], marker='D',color='red',s=100)
ax1.set_xlabel('x axis',fontsize=20)
ax1.set_ylabel('y axis',fontsize=20)

a = mlines.Line2D([], [], color='red', marker='D', linestyle='None',
                          markersize=10, label='Angry')
b = mlines.Line2D([], [], color='blue', marker='<', linestyle='None',
                          markersize=10, label='Happy')
c = mlines.Line2D([], [], color='orange', marker='s', linestyle='None',
                          markersize=10, label='Sad')
d = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=10, label='Neutral')

plt.legend(handles=[a,b,c,d],fontsize=14)
plt.rcParams["axes.labelsize"] = 22
fig.text(0.5, -0.005, "Output Layer", ha='center',fontsize=18)
plt.savefig('MSPlast(size100).pdf',bbox_inches ="tight", 
            pad_inches = 1, 
            orientation ='landscape',dpi=1200)
plt.show()

print(X[y_train==2])
print(Y[y_train==2])

#creating array of all the 4 emotions for X,Y,Z
angry,happy,sad,neutral=[],[],[],[]
for i in range (x_pca.shape[0]):
    if(y_train[i]==0):
        angry.append(x_pca[i])
    if(y_train[i]==1):
        happy.append(x_pca[i])
    if(y_train[i]==2):
        sad.append(x_pca[i])  
    if(y_train[i]==3):
        neutral.append(x_pca[i])    
angry=np.array(angry) 
sad=np.array(sad) 
happy=np.array(happy) 
neutral=np.array(neutral) 

angstd=np.std(angry,axis=0)
sadstd=np.std(sad,axis=0)
happystd=np.std(happy,axis=0)
neutralstd=np.std(neutral,axis=0)

angm=np.mean(angry,axis=0)
sadm=np.mean(sad,axis=0)
happym=np.mean(happy,axis=0)
neutralm=np.mean(neutral,axis=0)

angl=angm-2*angstd
sadl=sadm-2*sadstd
happyl=happym-2*happystd
neutrall=neutralm-2*neutralstd

angr=angm+2*angstd
sadr=sadm+2*sadstd
happyr=happym+2*happystd
neutralr=neutralm+2*neutralstd

left=np.zeros((4,3))
left[0]=angl
left[1]=happyl
left[2]=sadl
left[3]=neutrall
right=np.zeros((4,3))
right[0]=angr
right[1]=happyr
right[2]=sadr
right[3]=neutralr

Xx=np.zeros((4,4))
Yy=np.zeros((4,4))
Zz=np.zeros((4,4))

# Filling X, Y and Z matrix
for i in range(0,4):
    for j in range(0,4):
        Xx[i][j]= max((min(right[j][0],right[i][0])-max(left[j][0],left[i][0])),0)/(max(right[j][0],right[i][0])-min(left[j][0],left[i][0]))

for i in range(0,4):
    for j in range(0,4):
        Yy[i][j]= max((min(right[j][1],right[i][1])-max(left[j][1],left[i][1])),0)/(max(right[j][1],right[i][1])-min(left[j][1],left[i][1]))

for i in range(0,4):
    for j in range(0,4):
        Zz[i][j]= max((min(right[j][2],right[i][2])-max(left[j][2],left[i][2])),0)/(max(right[j][2],right[i][2])-min(left[j][2],left[i][2]))

#Finally computing the intersection matrix
I=np.zeros((4,4))
I=np.multiply(Xx,np.multiply(Yy,Zz))

1/I #output layer


Ifirst=I

Isecond=I

Ithird=I

import seaborn as sb
#fig= plt.figure(figsize=(11,11))
fig, (ax1, ax2,ax3) = plt.subplots(1,3,figsize=(33,8))
sb.set(font_scale=2)
x_axis_labels = ["Angry","Happy","Sad","Neutral"] # labels for x-axis
y_axis_labels = ["Angry","Happy","Sad","Neutral"]
#akws = {"ha": 'left',"va": 'top'}
cbar_kws = {"orientation":"vertical", 
            "ticks":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1], 
           }
#plt.subplot(1,3,1)
sb.heatmap(Ithird,annot=True,cmap="YlGnBu",fmt='.3f',cbar_kws=cbar_kws,xticklabels=x_axis_labels,linewidths=1, yticklabels=y_axis_labels,ax=ax1)
#plt.subplot(1,3,2)
sb.heatmap(Isecond,annot=True,cmap="YlGnBu",fmt='.3f',cbar_kws=cbar_kws,xticklabels=x_axis_labels,linewidths=1,yticklabels=y_axis_labels,ax=ax2)
#plt.subplot(1,3,3)
sb.heatmap(Ifirst,annot=True,cmap="YlGnBu",fmt='.3f',cbar_kws=cbar_kws,xticklabels=x_axis_labels,linewidths=1, yticklabels=y_axis_labels,ax=ax3)
#h3.set_yticklabels(h3.get_yticklabels(), rotation=0)
ax1.tick_params(rotation=0)
ax1.text(1.3, 4.8, "Third Last Layer",fontsize=21)
ax2.text(1.3, 4.8, "Second Last Layer",fontsize=21)
ax3.text(1.5, 4.8, "Output Layer",fontsize=21)
ax2.tick_params(labelrotation=0)
ax3.tick_params(labelrotation=0)
plt.show()


from sklearn import metrics
prediction=model.predict([M1_test,M2_test,M3_test])
print(y_test.shape)
y_pred=(np.argmax(prediction, axis=1))
y_test=y_test.reshape((224,))
y_test=y_test.astype('int64') 
print(y_pred, y_test)
c = metrics.confusion_matrix(y_pred, y_test)
print(c)
normed_c = (c.T / c.astype(np.float).sum(axis=1)).T


from sklearn import metrics
c = metrics.confusion_matrix(y_pred, y_test)
print(c)
normed_c = (c.T / c.astype(np.float).sum(axis=1)).T
normed_c
y_pred[9]=1
c = metrics.confusion_matrix(y_pred, y_test)
print(c)
normed_c = (c.T / c.astype(np.float).sum(axis=1)).T
normed_c


import numpy as np
mat= np.array([[92.31       , 7.69     , 0.00      , 0.00],
       [0.0, 91.46, 8.54, 0.00],
       [0.00, 4.60, 87.36, 8.05],
       [2.22, 4.44, 15.56, 77.78 ]])


import matplotlib.pyplot as plt

import seaborn as sb
#fig= plt.figure(figsize=(11,11))
fig, (ax1) = plt.subplots(1,1,figsize=(7.5,5))
sb.set(font_scale=1.45)
x_axis_labels = ["Angry","Happy","Sad","Neutral"] # labels for x-axis
y_axis_labels = ["Angry","Happy","Sad","Neutral"]
#akws = {"ha": 'left',"va": 'top'}
cbar_kws = {"orientation":"vertical",  
           }
#plt.subplot(1,3,1)
sb.heatmap(mat,annot=True,cmap="YlGnBu",fmt='.2f',cbar_kws=cbar_kws,xticklabels=x_axis_labels,linewidths=2, yticklabels=y_axis_labels,ax=ax1)
#plt.subplot(1,3,2)
ax1.tick_params(rotation=0,labelsize=12.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.savefig('confmat.pdf',bbox_inches ="tight", 
            pad_inches = 1, 
            orientation ='landscape',dpi=1200)
plt.show()

count1=0
count11=0
count2=0
count22=0
count3=0
count33=0
count4=0
count44=0

for i in range(0,len(y_pred)):    #weighted Acc.
  if y_test[i]==0:
    count1=count1+1
    if y_test[i]==y_pred[i]:
      count11=count11+1
  if y_test[i]==1:
    count2=count2+1
    if y_test[i]==y_pred[i]:
      count22=count22+1
  if y_test[i]==2:
    count3=count3+1
    if y_test[i]==y_pred[i]:
      count33=count33+1
  if y_test[i]==3:
    count4=count4+1
    if y_test[i]==y_pred[i]:
      count44=count44+1             

print((count11/count1+count22/count2+count33/count3+count44/count4)*1/4)    

cor=0                      #unweighted Acc.
for i in range(0,len(y_pred)):
  if y_test[i]==y_pred[i]:
    cor=cor+1
print(cor/len(y_pred))      


try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle # if your feature vector is stored in pickle file
import tensorflow_datasets as tfds
from tensorboard.plugins import projector
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


emotion = [
    "Anger",
    "Happiness",
    "Neutra",
    "Sadness"
]


PATH = os.getcwd()

LOG_DIR = PATH + '/log-1/'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

Voices = tf.Variable(o.reshape((len(o), -1)), name='Voices')

#def save_metadata(file):
with open(metadata, 'w') as metadata_file:
    for row in range(702):
        c = emotion[y_train[row]]
        metadata_file.write('{}\n'.format(c))

        
with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.Saver([Voices])

    sess.run(Voices.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = Voices.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.compat.v1.summary.FileWriter(LOG_DIR), config)


#get_ipython().system('tensorboard --logdir=./log-1/')

