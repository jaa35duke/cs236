from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, MaxPooling1D, SeparableConv1D, AveragePooling1D, Flatten
from keras import optimizers
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.metrics import *
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt


'''df = pd.read_csv('data/Model_Input_Data/_data_final_v2_Q_returns.csv', sep=',')
df = df.fillna(0)
X = df.iloc[:, 7:-1]
x = X.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
#X = pd.DataFrame(x_scaled)
yT = df.iloc[:, -1]
n = len(yT)
y = pd.Series([])
k = 523
inf = 100
for i in range(0, int(n/k)):
    vals = yT[i*k:(i+1)*k]
    yCat = pd.qcut(vals, 4, labels=False)
    y = pd.concat([y, yCat])

n_c = 523
n_t = int(n/k)
y_new = np.zeros((n_c, 1))
#X_new = np.zeros((n_c, len(X.keys()), 1, n_t))
X_new = np.zeros((n_c, len(X.keys()), n_t))
for c in range(0, n_c):
    vals = x_scaled[c::n_c]
    #vals = np.transpose(np.expand_dims(vals, axis=1))
    vals = np.transpose(vals)
    X_new[c] = vals
    y_new[c] = y[c + (n_t-1)*n_c]

y = np_utils.to_categorical(y_new)

np.save('cnn_y', y)
np.save('cnn_X', X_new)
'''

y = np.load('cnn_y.npy')
X = np.load('cnn_X.npy')

X = np.swapaxes(X, 1, 2)

X_trainDev, X_test, y_trainDev, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_dev, y_train, y_dev = train_test_split(X_trainDev, y_trainDev, test_size=0.25, random_state=1)

drop = 0.5 #fraction of input units to set to 0
y_dim = y.shape[1]
batch_size = 256
epochs = 100
learning_rate = 0.01

model = Sequential()
model.add(Conv1D(128, kernel_size=1, strides=1, padding='valid', dilation_rate=1, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(drop))
model.add(Conv1D(64, kernel_size=1, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(drop))
#model.add(AveragePooling1D(3))
model.add(SeparableConv1D(16, kernel_size=1, data_format='channels_last', depth_multiplier=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(drop))
#model.add(AveragePooling1D(2))
model.add(Flatten())
#model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(drop))
model.add(Dense(y_dim, activation='softmax'))
print(model.summary())

opt = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
#opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs, batch_size=batch_size, verbose=2)

score_train = model.evaluate(X_train, y_train, batch_size=batch_size)
score_dev = model.evaluate(X_dev, y_dev, batch_size=batch_size)
print("Train accuracy: " + str(score_train))
print("Dev accuracy: " + str(score_dev))

pred_dev = model.predict(X_dev)
pred_dev = np_utils.to_categorical(np.argmax(pred_dev, axis=-1))
print("Classification report: " + str(classification_report(y_dev, pred_dev)))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Dev'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Dev'], loc='upper left')
plt.show()



