# Import libraries 
import sys
sys.path.insert(1,'/home/pawan/CREMA/')

import librosa
import seaborn as sns
import librosa.display
import numpy as np
from features import *
import matplotlib.pyplot as plt
#import tensorflow as tf
from matplotlib.pyplot import specgram
from scipy.stats import kurtosis, skew
import pandas as pd
import glob 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import os
from sklearn import svm
import sys
import pickle
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split



RAV = "/home/pawan/CREMA"
CREMA = "/home/pawan/CREMA/wav_files/"


## LOAD CREMA dataset

tr = pd.read_csv('/home/pawan/CREMA/VideoDemographics.csv')         

dir_list = os.listdir(CREMA)
dir_list.sort()
gender = []
race = []
path = []


spkr_list = [1002,1003, 1011, 1014, 1022, 1023, 1026, 1045, 1019, 1072, 1081, 1085, 1090, 1091,1073,1074, 1070, 1059, 1050,1042, 1038]

for i in dir_list: 
    part = i.split('_')

    trA = tr.Race.loc[tr.ActorID == int(part[0])].values
    if int(part[0]) in spkr_list :   
       gender.append('CREMA')
       race.append(trA[0])  #temp + '_' + 
       path.append(CREMA + i)
    
CREMA_df = pd.DataFrame(race, columns = ['labels'])
CREMA_df['source'] = 'CREMA'
CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)
CREMA_df.labels.value_counts()

df = pd.concat([CREMA_df], axis = 0)
df.to_csv("/home/pawan/CREMA/Data_path.csv",index=False)


# lets pick up the meta-data that we got from our first part of the Kernel
ref = pd.read_csv("/home/pawan/CREMA/Data_path.csv")


# Note this takes a couple of minutes (~10 mins) as we're iterating over 4 datasets 
df = pd.DataFrame(columns=['feature'])

# loop feature extraction over the entire dataset
print('EXTRACTING SPECTRAL FEATURES')

counter=0

for index,path in enumerate(ref.path):

    X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=2.5, sr=16000, offset=0.5)
    sample_rate = np.array(sample_rate)
    
    # compute the features frame-wise
    coeff = mfcc(X, sample_rate, 0.025, 0.010, 20)
    
    # Retain features of voiced frames (Based on root mean square energy)
    energy = librosa.feature.rms(y=X, frame_length = 400, hop_length = 160).reshape(-1)

    temp = []
    
    for i in range(len(energy)-2):
        if energy[i] > 0.09 * max(energy):
           temp.append(coeff[i,:])
           
    coeff = np.array(temp)
    

    feats = np.hstack((np.hstack((np.mean(coeff,0), np.hstack((np.std(coeff, 0), np.median(coeff, 0))))), np.hstack((skew(coeff,0), kurtosis(coeff, 0))) ))

    
    df.loc[counter] = [feats]
    counter=counter+1   

#print(coeff.shape)

# Now extract the mean bands to its own feature columns
df = pd.concat([ref, pd.DataFrame(df['feature'].values.tolist())],axis=1)


# replace NA with 0
df=df.fillna(0)


# ------------------------------------------------------------------------------------------------------------
# 					x`TRAINING CLASSIFIER
#-------------------------------------------------------------------------------------------------------------



print('TRAINING SVM CLASSIFIER......................')

i = 0


n_iter = 1
acc = [0] * n_iter

for j in range(1):
   for i in range(n_iter):
       # Split between train and test 
       X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','labels','source'],axis=1), df.labels, test_size=0.25, shuffle=True, random_state=42)
       scaler = StandardScaler()  # doctest: +SKIP
       scaler.fit(X_train)  # doctest: +SKIP
       X_train = scaler.transform(X_train)  # doctest: +SKIP
       X_test = scaler.transform(X_test)

       clf = svm.SVC(kernel ='rbf', C = 10, gamma = 0.01).fit(X_train, y_train)
       y_pred = clf.predict(X_test)
       acc[i] = accuracy_score(y_test, y_pred) * 100

print('The mean accuracy on validation data is:', np.mean(acc),'+/-', np.std(acc))

#unique, counts = np.unique(y_test, return_counts=True)
#print(np.asarray((unique, counts)).T)

plot_confusion_matrix(clf, X_test, y_test, normalize= 'true')
plt.show()

