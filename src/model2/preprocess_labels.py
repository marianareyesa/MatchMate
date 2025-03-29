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
from keras.layers import Dropout, Dense, GRU
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers
from tqdm import tqdm

####################################################################

X = []
y = []

folders = ["andrea_shots"]

KEYPOINTS_TO_KEEP = [
    "nose_y", "nose_x",
    "left_shoulder_y", "left_shoulder_x",
    "right_shoulder_y", "right_shoulder_x",
    "left_elbow_y", "left_elbow_x",
    "right_elbow_y", "right_elbow_x",
    "left_wrist_y", "left_wrist_x",
    "right_wrist_y", "right_wrist_x",
    "left_hip_y", "left_hip_x",
    "right_hip_y", "right_hip_x",
    "left_knee_y", "left_knee_x",
    "right_knee_y", "right_knee_x",
    "left_ankle_y", "left_ankle_x",
    "right_ankle_y", "right_ankle_x"
]

def normalize_keypoints(sequence):
    x = sequence[:, 0::2]
    y = sequence[:, 1::2]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x = (x - x_min) / (x_max - x_min + 1e-6)
    y = 1 - ((y - y_min) / (y_max - y_min + 1e-6))  # Flip Y

    sequence[:, 0::2] = x
    sequence[:, 1::2] = y
    return sequence


WINDOW_SIZE = 30  # length of each input sequence
X = []
y = []

for folder in folders:
    folder_path = f"/Users/marianareyes/Desktop/MatchMate/MatchMate/dataset/{folder}/"
    if not os.path.exists(folder_path):
        print(f"{folder_path} doesn't exist")
        continue

    print(f"Loading shots from {folder_path}")

    for shot_csv in tqdm(sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])):
        data = pd.read_csv(os.path.join(folder_path, shot_csv))
        data = data[KEYPOINTS_TO_KEEP + ["shot"]]

        # Step 1: group consecutive rows with same shot label
        prev_label = None
        buffer = []

        for _, row in data.iterrows():
            current_label = row["shot"]

            if current_label != prev_label and buffer:
                # process buffered group before starting a new one
                for i in range(0, len(buffer) - WINDOW_SIZE + 1, WINDOW_SIZE):
                    window = np.array(buffer[i:i+WINDOW_SIZE])
                    norm_window = normalize_keypoints(window.copy())
                    X.append(norm_window)
                    y.append(prev_label)

                buffer = []

            keypoints = row.drop("shot").to_numpy()
            buffer.append(keypoints)
            prev_label = current_label

        # process last group
        for i in range(0, len(buffer) - WINDOW_SIZE + 1, WINDOW_SIZE):
            window = np.array(buffer[i:i+WINDOW_SIZE])
            norm_window = normalize_keypoints(window.copy())
            X.append(norm_window)
            y.append(prev_label)

X = np.stack(X, axis=0)
y = np.array(y)
X = np.array(X)

print(f"Loaded {len(y)} shots for training")

# TEMP: Use the same data as test to keep format
X_test = X.copy()
y_test = y.copy()

print(f"Loaded {len(y_test)} shots for test")

####################################################################

shots = list(set(y))
occurences = [np.count_nonzero(y == shot) for shot in shots]
print(shots, occurences)

fig, ax = plt.subplots()
ax.pie(occurences, labels=shots, autopct='%1.1f%%')
ax.axis('equal')
ax.set_title('Shots dataset')

####################################################################

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, shuffle=True)

print(f"Shape of train features : {X_train[0].shape}")
print(f"Shape of val features : {X_val[0].shape}")

print("Total categories: ", len(np.unique(y_train)))
print("Total categories: ", len(np.unique(y_val)))

nb_cat = len(np.unique(y_train))


####################################################################

le = preprocessing.LabelEncoder()
le.fit(np.concatenate((y_train, y_val, y_test)))
nb_cat = len(le.classes_)


y_train = le.transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)


y_train = tf.keras.utils.to_categorical(y_train, num_classes=nb_cat)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=nb_cat)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=nb_cat)

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

assert len(le.classes_) == nb_cat

print("X_train Shape: ", X_train.shape)
print("X_val Shape: ", X_val.shape)
print("X_test Shape: ", X_test.shape)
print("y_train Shape: ", y_train.shape)
print("y_val Shape: ", y_val.shape)
print("y_test Shape: ", y_test.shape)

####################################################################

m1 = Sequential()
m1.add(GRU(units=24, dropout=0.1, input_shape=(30, 26)))
m1.add(Dropout(0.2))
m1.add(Dense(units=8, activation='relu'))
m1.add(Dense(units=nb_cat, activation='softmax'))

m1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
m1.summary()

X_train = np.array(X_train).astype(np.float32)
X_val = np.array(X_val).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)

filepath = "weights_feedback.keras"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=False, save_best_only=True)
hist = m1.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              batch_size=32,
              epochs=300,
              verbose=1,
              callbacks=[checkpointer])

loss, accuracy = m1.evaluate(X_val, y_val)
print(f"Accuracy on validation dataset = {accuracy}")

m1.load_weights(filepath)
loss, accuracy = m1.evaluate(X_val, y_val)
print(f"Accuracy on validation dataset = {accuracy}")

m1.save("tennis_rnn_feedback.keras")

####################################################################

loss, accuracy = m1.evaluate(X_test, y_test)
print(f"Accuracy on test dataset = {accuracy:.3f}")

preds = m1.predict(X_test)
test_predictions = np.argmax(preds, axis=1)

cm = confusion_matrix(y_true=np.argmax(y_test, axis=1), y_pred=test_predictions)
df_cm = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
plt.figure(figsize=(20, 20))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='.4g')
plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=20)
plt.savefig("confusion_matrix_feedback.png")

####################################################################

def to_gif(shot):
    height = 500
    width = 500

    KEYPOINTS = np.array([
        "nose",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ])

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

    frames = np.zeros((30, height, width, 3), np.uint8)
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

k = random.randint(0, len(y_test) - 1)

converted_images = to_gif(X_test[k, :, :])
imageio.mimsave("animation_feedback.gif", converted_images, fps=15)

print("Predicts:")
for i in range(nb_cat):
    print(f"{le.classes_[i]} = {preds[k, i]*100:.1f}%")

print("DONE")