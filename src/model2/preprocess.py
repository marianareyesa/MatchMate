import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random

import imageio

from sklearn.metrics import confusion_matrix
import seaborn as sn

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, GRU
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers


from tqdm import tqdm

####################################################################

X=[]
y=[]

folders = ["alcaraz", "dimitrov_thiem", "djoko_sock", "federer", "nadal", "roland"]

for folder in folders:
    if not os.path.exists(f"/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset2/{folder}/shots/"):
        # /Users/marianareyes/Desktop/MatchMate/MatchMate/dataset2
        print(f"dataset/{folder}/shots/ doesnt exist")
        continue
        
    print(f"Loading shots from dataset/{folder}/shots/")
        
    for shot_csv in tqdm(sorted(os.listdir(f'/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset2/{folder}/shots/'))):
        data = pd.read_csv(os.path.join(f'/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset2/{folder}/shots/', shot_csv))
        
        if folder == "nadal":
            revert_data= data.copy()
            for feature in data.columns:
                if feature[-2:]=="_x":
                    revert_data[feature] = 1 - data[feature]
            data = revert_data

        features = data.loc[:, data.columns != 'shot']

        X.append(features.to_numpy())
        y.append(data["shot"].iloc[0])
    
X = np.stack(X, axis=0)

y = np.array(y)
X = np.array(X)

print(f"Loaded {len(y)} shots for training")

X_test=[]
y_test=[]

folders = ["roland", "alcaraz"]
for folder in folders:
    for shot_csv in sorted(os.listdir(f'/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset2/{folder}/shots/')):
    
        data = pd.read_csv(os.path.join(f'/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset2/{folder}/shots/',shot_csv ))

        if folder == "dataset_nadal":
                revert_data= data.copy()
                for feature in data.columns:
                    if feature[-2:]=="_x":
                        revert_data[feature] = 1 - data[feature]
                data = revert_data


        features = data.loc[:, data.columns != 'shot']

        X_test.append(features.to_numpy())
        y_test.append(data["shot"].iloc[0])
    
X_test = np.stack(X_test, axis=0)

y_test = np.array(y_test)
X_test = np.array(X_test)

print(f"Loaded {len(y_test)} shots for test")

####################################################################

shots = list(set(y))
occurences = [np.count_nonzero(y == shot) for shot in shots]
print(shots, occurences)

fig, ax = plt.subplots()
ax.pie(occurences, labels=shots, autopct='%1.1f%%')
ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
ax.set_title('Shots dataset')

#plt.savefig("shots_dataset.png")

####################################################################

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle= True)


print(f"Shape of train features : {X_train[0].shape}")
print(f"Shape of val features : {X_val[0].shape}")

print("Total categories: ", len(np.unique(y_train)))
print("Total categories: ", len(np.unique(y_val)))

nb_cat = len(np.unique(y_train))

####################################################################

le = preprocessing.LabelEncoder()

y_train = le.fit_transform(y_train)
y_val = le.fit_transform(y_val)
y_test = le.fit_transform(y_test)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=nb_cat)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=nb_cat)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=nb_cat)

y_train = np.array(y_train)
X_train = np.array(X_train)
#X_train = np.expand_dims(X_train, axis=-1)

y_val = np.array(y_val)
X_val = np.array(X_val)
#X_val = np.expand_dims(X_val, axis=-1)

y_test = np.array(y_test)
X_test = np.array(X_test)

assert len(le.classes_) == nb_cat

print("X_train Shape: ", X_train.shape)
print("X_val Shape: ", X_val.shape)
print("X_test Shape: ", X_test.shape)
print("y_train Shape: ", y_train.shape)
print("y_val Shape: ", y_val.shape)
print("y_test Shape: ", y_test.shape)

####################################################################

m1=Sequential()
m1.add(GRU(units=24, dropout=0.1, input_shape=( 30, 26)))
m1.add(Dropout(0.2))
m1.add(Dense(units = 8, activation = 'relu'))
m1.add(Dense(units = nb_cat, activation = 'softmax'))

m1.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
m1.summary()

filepath = "weights.keras"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=False, save_best_only=True)
hist = m1.fit(X_train, y_train,
               validation_data=(X_val, y_val),
               batch_size = 32,
                epochs=300, 
                verbose = 1, 
                callbacks=[checkpointer]
             )

loss, accuracy = m1.evaluate(X_val, y_val)
print(f"Accuracy on validation dataset = {accuracy}")

m1.load_weights(filepath)

loss, accuracy = m1.evaluate(X_val, y_val)
print(f"Accuracy on validation dataset = {accuracy}")

#m1.save("tennis_rnn.keras")

####################################################################
loss, accuracy = m1.evaluate(X_test, y_test)
print(f"Accuracy on test dataset = {accuracy:.3f}")

preds = m1.predict(X_test)
test_predictions = np.argmax(preds, axis=1)

cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=test_predictions)


df_cm = pd.DataFrame(
    cm, index=le.classes_, columns=le.classes_
)
plt.figure(figsize=(20, 20))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.4g')
plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=20)
#plt.savefig("confusion_matrix.png")

####################################################################

def to_gif(shot):
    height = 500
    width = 500
    
    KEYPOINTS = np.array(["nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle"])
    
    EDGES = {
        ("nose", "left_shoulder"): "m",
        ("nose", "right_shoulder"): "c",
        ("left_shoulder", "left_elbow"): "m",
        ("left_elbow", "left_wrist"): "m",
        ("right_shoulder", "right_elbow"): "c",
        ("right_elbow", "right_wrist"): "c",
        ("left_shoulder", "right_shoulder"): "y",
        ("left_shoulder", "left_hip"): "m",
        ("right_shoulder", "right_hip"): "c",
        ("left_hip", "right_hip"): "y",
        ("left_hip", "left_knee"): "m",
        ("left_knee", "left_ankle"): "m",
        ("right_hip", "right_knee"): "c",
        ("right_knee", "right_ankle"): "c",
    }
    
    COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}
    
    
    frames=np.zeros((30, height, width, 3), np.uint8)
    for i in range(len(shot)):
        shot_inst = shot[i, :]
        for k in range(13):
            cv2.circle(
                frames[i],
                (
                    int(shot_inst[2*k+1] * width),
                    int(shot_inst[2*k] * height),
                ),
                radius=3,
                color=(0, 255, 0),
                thickness=-1,
            )
            
        for edge in EDGES.items():
            k = np.argwhere(KEYPOINTS == edge[0][0])[0][0]
            j = np.argwhere(KEYPOINTS == edge[0][1])[0][0]
            cv2.line(
                frames[i],
                (
                    int(shot_inst[2*k+1] * width),
                    int(shot_inst[2*k] * height),
                ),
                (
                    int(shot_inst[2*j+1] * width),
                    int(shot_inst[2*j] * height),
                ),
                color=COLORS[edge[1]],
                thickness=2,
            )
    
    return frames.astype(np.uint8)

k = random.randint(0, len(y_test))

converted_images = to_gif(X_test[k, :, :])
imageio.mimsave("animation.gif", converted_images, fps=15)

print("Predicts:")
for i in range(nb_cat):
    print(f"{le.classes_[i]} = {preds[k, i]*100:.1f}%")

print("DONE")
