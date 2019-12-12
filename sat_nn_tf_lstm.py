from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, LSTM, SimpleRNN
from keras import optimizers
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn import preprocessing

import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt

rand = 123

df = pd.read_csv('data/Model_Input_Data/_data_final_v2_Q_returns.csv', sep=',')
df = df.fillna(0)
X = df.iloc[:, 7:-1]

x = X.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)

yT = df.iloc[:, -1]
n = len(yT)
y = pd.Series([])
k = 523
inf = 100
y_dim = 4
for i in range(0, int(n/k)):
    vals = yT[i*k:(i+1)*k]
    mu, std = norm.fit(vals)
    #bins = [-inf, mu - 2*std, mu + 2*std, inf]
    #bins = [-inf, mu - 2 * std, mu - std, mu, mu + std, mu + 2 * std, inf]
    #y_dim = len(bins) - 1
    #yCat = pd.cut(vals, bins=bins, labels=False)
    yCat = pd.qcut(vals, y_dim, labels=False)
    y = pd.concat([y, yCat])

y = np_utils.to_categorical(y)

X_trainDev, X_test, y_trainDev, y_test = train_test_split(X, y, test_size=0.2, random_state=rand)
X_train, X_dev, y_train, y_dev = train_test_split(X_trainDev, y_trainDev, test_size=0.25, random_state=rand)


# create model
drop = 0.3   #fraction of input units to set to 0
epochs = 20
batch_size = 128
learning_rate = 0.05
alpha = 0.3

model = Sequential()
n_cols = len(X_train.keys())

model = Sequential()
model.add(Embedding(1030, 128, input_length=n_cols))
#model.add(LSTM(output_dim=32, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Dropout(drop))

model.add(SimpleRNN(64, activation='tanh'))

model.add(Dense(16, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=alpha))
model.add(Dropout(drop))

model.add(Dense(y_dim, kernel_initializer='uniform'))
model.add(Activation('softmax'))

#opt = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
history = model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])

history = model.fit(X_train, y_train,
          validation_data=(X_dev, y_dev),
          epochs=epochs,
          batch_size=batch_size,
          verbose=2)



score_train = model.evaluate(X_train, y_train, batch_size=batch_size)
score_dev = model.evaluate(X_dev, y_dev, batch_size=batch_size)
print("Train accuracy: " + str(score_train))
print("Dev accuracy: " + str(score_dev))

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



"""model.add(Dense(128, input_dim=len(X_train.keys()), init='uniform'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=alpha))
model.add(Dropout(drop))

model.add(Dense(32, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=alpha))
model.add(Dropout(drop))

model.add(Dense(16, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=alpha))
model.add(Dropout(drop))

model.add(Dense(8, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=alpha))
model.add(Dropout(drop))

model.add(Dense(y_dim, kernel_initializer='uniform'))
model.add(Activation('softmax'))"""