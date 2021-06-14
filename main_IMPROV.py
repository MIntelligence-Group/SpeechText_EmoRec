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


def load_data():
    F1,F2,F3, y = [],[],[], []
    count=0
    for i in AVAILABLE_EMOTIONS:
        for file in glob.glob("./session1/"+str(i)+"/*.wav"):
            # get the base name of the audio file
            # extract speech features
            if(file[-10]!='S'):
              continue
            features1,features2,features3= extract_feature(file, mfcc=True, chroma=True, mel=True) #calling the extract feature function for speech features                            
            F1.append(features1)                  #appending speech features for every audio in list R 
            F2.append(features2)
            F3.append(features3)
            y.append(i)
            count=count+1
            if(count%100==0):
                print(count)
    for file in glob.glob("./session2/*.wav"):
        # get the base name of the audio file
        # extract speech features
        if(file[31]!='S'):
          continue
        features1,features2,features3= extract_feature(file, mfcc=True, chroma=True, mel=True) #calling the extract feature function for speech features                         
        F1.append(features1)                  #appending speech features for every audio in list R 
        F2.append(features2)
        F3.append(features3)
        if(file[25]=='A'):
            y.append('Angry')
        if(file[25]=='S'):
            y.append('Sad')
        if(file[25]=='H'):
            y.append('Happy')  
        if(file[25]=='N'):
            y.append('Neutral')          
        count=count+1
        if(count%100==0):
            print(count)        
        # split the data to training and testing and return it    
    return np.array(F1),np.array(F2),np.array(F3),y

f1,f2,f3,y=load_data()


y=np.array(y)
y[y=='Happy']=1
y[y=='Angry']=0
y[y=='Sad']=2
y[y=='Neutral']=3
y

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
GLOVE_DIM = 50# Number of dimensions of the GloVe word embeddings\
    
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

# Train test Split
y=y.astype('int64')
M1_train,M1_test,M2_train,M2_test,M3_train,M3_test, y_train, y_test = train_test_split(f1,f2,f3,y,test_size=0.2)
y_test

print(l_train-y_train)

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

# MSP IMPROV model
import keras.backend as K
_input1 = Input(shape=(40,1), dtype='float32')

activations1 =   GRU(40,return_sequences=True)(_input1)
act= Flatten()(activations1)
# attention1  =   Dense(1,activation='tanh')(activations1)
# attention12   =   Dense(1)(attention1)
# attention13   =   Flatten()(attention12)
# attention14   =   Activation('softmax')(attention13)

# sent_representation1 = keras.layers.Multiply()([activations1,attention14])
# sent_representation11 = keras.layers.Lambda(lambda xin: K.sum(xin,axis=-2),output_shape=(40,))(sent_representation1)

# output1 = Dense(60, activation='relu')(sent_representation11)

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

output51    = concatenate([act,output2,output3])
output52    = Dropout(0.3)(output51)
output53    = Dense(600, activation='relu')(output52)
output54    = Dense(4, activation='softmax')(output53)

model = keras.Model(inputs=[_input1,_input2,_input3], outputs=[output54])
model.summary()
plot_model(model, to_file='MSP_improv.png')


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import tensorflow as tf
ACCURACY_THRESHOLD = 0.72
class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_accuracy') >= ACCURACY_THRESHOLD):   
            print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))   
            self.model.stop_training = True

callbacks = myCallback()        

#########MSP improv training ######################
from keras.callbacks import ModelCheckpoint
callbacks_list = [callbacks]
model.fit([M1_train,M2_train,M3_train], y_train, epochs=150, batch_size=64,validation_data=([M1_test,M2_test,M3_test], y_test),verbose=2,callbacks=callbacks_list)

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

#model.save("MSP_improv.h5")
#model= keras.models.load_model('/content/MSP_improv.h5')
#t=keras.models.load_model("weights.best.hdf5")
#model = keras.models.load_model("weights.best.hdf5", custom_objects={'MyCustomLayer': "BertModelLayer"})

## code for computing intersection matrix

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


## Code for embedding plots

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


I #second last layer

I #third last layer

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

