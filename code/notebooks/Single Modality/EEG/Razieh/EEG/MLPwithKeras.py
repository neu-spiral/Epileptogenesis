import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

df = scipy.io.loadmat('data/EVENTS.mat')
df = df['EVENTS']

# Separate majority and minority classes
df_majority0 = df[df[:, -1] == 0]
df_majority1 = df[df[:, -1] == 1]
df_majority3 = df[df[:, -1] == 3]

df_minority2 = df[df[:, -1] == 2]#550

# Downsample majority class
df_majority0_downsampled = resample(df_majority0,
                                   replace=False,  # sample without replacement
                                   n_samples=550,  # to match minority class
                                   random_state=123)  # reproducible results

df_majority1_downsampled = resample(df_majority1,
                                   replace=False,  # sample without replacement
                                   n_samples=550,  # to match minority class
                                   random_state=123)  # reproducible results

df_majority3_downsampled = resample(df_majority3,
                                   replace=False,  # sample without replacement
                                   n_samples=550,  # to match minority class
                                   random_state=123)  # reproducible results

df_downsampled = np.concatenate([df_majority0_downsampled, df_majority1_downsampled, df_majority3_downsampled,\
                            df_minority2])
np.random.shuffle(df_downsampled)

features_numpy = df_downsampled[:, :-1]
targets_numpy = df_downsampled[:, -1]

# np.random.shuffle(df)
# data = df[:,:-1]
# target = df[:,-1]
# target[target != 0] = 1

# 2. standardize the data such that it has mean of 0 and standard deviation of 1
mean = features_numpy.mean(axis=0)
features_numpy -= mean
std = features_numpy.std(axis=0)
features_numpy /= std

model = Sequential()
model.add(layers.Dense(132, activation="relu", kernel_regularizer=regularizers.l1(0.001), input_shape=(1250,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l1(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()


# The train_test_split function from sklearn helps us to create training and test sets.
x_train, x_test, y_train, y_test = train_test_split(features_numpy, targets_numpy, test_size=0.2, random_state=42)

# compile the model and train it for 100 epochs.
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(x_train, y_train, epochs=500, batch_size=64, validation_split=0.2, verbose=2)

# we will unleash the model on the test set, plot the ROC curve and calculate the AUC
y_pred = model.predict(x_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
AUC = auc(fpr_keras, tpr_keras)
plt.plot(fpr_keras, tpr_keras, label='Keras Model(area = {:.3f})'.format(AUC))
plt.xlabel('False positive Rate')
plt.ylabel('True positive Rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

