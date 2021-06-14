import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.metrics import accuracy_score # to measure how good we are

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, required=True)    
    args = parser.parse_args()
    return args

args = parse_args()

epochs = args.epoch

def extract_feature(file_name, **kwargs):
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
        result = np.array([])
        if mfcc:
            mfccs =  np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
            #print(mfccs.shape)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
            #print(chroma.shape)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
            #print(mel.shape)
    return result


int2emotion = {
    "1": "neutral",
    "2": "calm",
    "3": "happy",
    "4": "sad",
    "5": "angry",
    "6": "fearful",
    "7": "disgust",
    "8": "surprised"
}

AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "happy",
    "neutral",
}


def load_data():
    X, y = [], []
    count=0
    for file in glob.glob("emo*/*.wav"):
        # get the base name of the audio file
        basename = os.path.split(file)[0]
        # get the emotion label
        emotion = int2emotion[basename[4]]
        
        # we allow only AVAILABLE_EMOTIONS we set
        #print(emotion)
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # extract speech features
        features = extract_feature(file, mfcc=True, chroma=True, mel=True)
        # add to data
        X.append(features)
        y.append(emotion)
        count=count+1
        if(count%100==0):
            print(count)
    # split the data to training and testing and return it    
    return np.array(X),np.array(y)

M,l=load_data()


print(M.shape)
X_train, X_test, y_train, y_test = train_test_split(M,l,test_size=0.25,random_state=4)
l


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Activation
from keras.layers import Flatten

model = Sequential()
model.add(Dense(318, input_dim=180))
model.add(Dropout(0.325))
model.add(Activation('relu'))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.25))

#model.add(Dense(550, activation='relu'))
model.add(Dense(4, activation='softmax'))

model1 = Sequential()
model1.add(Dense(318, input_dim=180))
model1.add(Dropout(0.325))
model1.add(Activation('relu'))
model1.add(Dense(400, activation='relu'))
model1.add(Dropout(0.25))
model1.add(Dense(500, activation='relu'))
model1.add(Dropout(0.25))
#model.add(Dense(550, activation='relu'))
model1.add(Dense(4, activation='softmax'))

model2 = Sequential()
model2.add(Dense(318, input_dim=180))
model2.add(Dropout(0.325))
model2.add(Activation('relu'))
model2.add(Dense(400, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(500, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(600, activation='relu'))
model2.add(Dropout(0.25))
#model.add(Dense(550, activation='relu'))
model2.add(Dense(4, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
y_train[y_train=='happy']=1
y_train[y_train=='angry']=0
y_train[y_train=='sad']=2
y_train[y_train=='neutral']=3
#y_train[y_train=='calm']=1
#y_train[y_train=='surprised']=5
y_test[y_test=='happy']=1
y_test[y_test=='angry']=0
y_test[y_test=='sad']=2
y_test[y_test=='neutral']=3
#y_test[y_test=='calm']=1
#y_test[y_test=='surprised']=5
y_test=y_test.astype('int64') 
y_train=y_train.astype('int64') 
y_test

history=model.fit(X_train, y_train, epochs=100, batch_size=64,validation_data=(X_test, y_test),verbose=2)
#history1=model1.fit(X_train, y_train, epochs=100, batch_size=64,validation_data=(X_test, y_test),verbose=2)
#history2=model2.fit(X_train, y_train, epochs=100, batch_size=64,validation_data=(X_test, y_test),verbose=2)

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy on test data: %.2f' % (accuracy*100))
_, accu = model.evaluate(X_train, y_train)
print('Accuracy on train data: %.2f' % (accu*100))

model.save("model.h5")


_, accuracy = model2.evaluate(X_test, y_test)
print('Accuracy on test data: %.2f' % (accuracy*100))
_, accu = model2.evaluate(X_train, y_train)
print('Accuracy on train data: %.2f' % (accu*100))

model.save("model2.h5")

import keras
l=keras.models.load_model("Ravdess.h5")

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

from sklearn import metrics
prediction=model.predict(X_train)
y_pred=(np.argmax(prediction, axis=1))
y_test=y_test.reshape((305,))
y_test=y_test.astype('int64') 
matrix = metrics.confusion_matrix(y_train,y_pred)
matrix


# # code for computing intersection matrix
import tensorflow as tf
import keras.backend as K
out_layer=model.layers[4].output
output_fn= K.function([model.input],out_layer)
o=output_fn([X_train])#1st argument of the .fit() function
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

fig= plt.figure(figsize=(11,11))
ax=plt.axes()
ax.scatter(X,Y,c=y_train,marker='^')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')

plt.show()

angry,happy,sad,neutral=[],[],[],[]
for i in range (x_pca.shape[0]):
    if(y_train[i]==3):
        neutral.append(x_pca[i])
    if(y_train[i]==0):
        angry.append(x_pca[i])
    if(y_train[i]==1):
        happy.append(x_pca[i])  
    if(y_train[i]==2):
        sad.append(x_pca[i])    
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


for i in range(0,4):
    for j in range(0,4):
        Xx[i][j]= max((min(right[j][0],right[i][0])-max(left[j][0],left[i][0])),0)/(max(right[j][0],right[i][0])-min(left[j][0],left[i][0]))

for i in range(0,4):
    for j in range(0,4):
        Yy[i][j]= max((min(right[j][1],right[i][1])-max(left[j][1],left[i][1])),0)/(max(right[j][1],right[i][1])-min(left[j][1],left[i][1]))

for i in range(0,4):
    for j in range(0,4):
        Zz[i][j]= max((min(right[j][2],right[i][2])-max(left[j][2],left[i][2])),0)/(max(right[j][2],right[i][2])-min(left[j][2],left[i][2]))

I=np.zeros((4,4))
I=np.multiply(Xx,np.multiply(Yy,Zz))
I

1/I #output layer

1/I #third last layer

1/I #second last layer

matrix

Isecond=I

Ithird=I

Ifirst=I

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
ax1.text(1.3, 4.8, "Third Last Layer",fontsize=22)
ax2.text(1.3, 4.8, "Second Last Layer",fontsize=22)
ax3.text(1.5, 4.8, "Output Layer",fontsize=22)
ax2.tick_params(labelrotation=0)
ax3.tick_params(labelrotation=0)
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
plt.savefig('PCA_Output_Layer.pdf',bbox_inches ="tight", 
            pad_inches = 1, 
            orientation ='landscape',dpi=1200)
plt.show()

from sklearn import metrics
prediction=l.predict([X_test])
y_pred=(np.argmax(prediction, axis=1))
y_test=y_test.reshape((305,))
y_test=y_test.astype('int64') 
c = metrics.confusion_matrix(y_pred, y_test)
normed_c = (c.T / c.astype(np.float).sum(axis=1)).T
normed_c=normed_c*100

import seaborn as sb
#fig= plt.figure(figsize=(11,11))
fig, (ax1) = plt.subplots(1,1,figsize=(6,4))
sb.set(font_scale=1)
x_axis_labels = ["Angry","Happy","Sad","Neutral"] # labels for x-axis
y_axis_labels = ["Angry","Happy","Sad","Neutral"]
#akws = {"ha": 'left',"va": 'top'}
cbar_kws = {"orientation":"vertical",  
           }
#plt.subplot(1,3,1)
sb.heatmap(normed_c,annot=True,cmap="YlGnBu",fmt='.2f',cbar_kws=cbar_kws,xticklabels=x_axis_labels,linewidths=1, yticklabels=y_axis_labels,ax=ax1)
#plt.subplot(1,3,2)
ax1.tick_params(rotation=0,labelsize=11.5)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.savefig('confmat.pdf',bbox_inches ="tight", 
            pad_inches = 1, 
            orientation ='landscape',dpi=1200)
plt.show()

#!tensorboard --logdir=log-1/

