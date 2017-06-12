import numpy as np
import keras
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam, RMSprop
import zarr
from zarr import blosc
blosc.set_nthreads(18)

from normalization import LayerNormalization
from multi_gpu import make_parallel


batch_size = 512 
epochs = 3 

BCL = 40
eps = 50
pop_size = 10000

print("Loading data")
sol = zarr.open_array('/mnt/ORD_BCL_50_eps_100.zarr', mode='r', shape=(pop_size*eps*BCL*10, 42), chunks=(1024, None), dtype='float32', fill_value=0.)
sol_dt = zarr.open_array('/mnt/ORD_dt_BCL_50_eps_100.zarr', mode='r', shape=(pop_size*eps*BCL*10, 41), chunks=(1024, None), dtype='float32', fill_value=0.)

print("Splitting sets")
#split = np.random.permutation(sol.shape[0])
#train = split[80:]
#test = split[:20]
split = int(0.8 * sol.shape[0])
#x_train = sol[:split, :]
#y_train = sol_dt[:split, :]
#x_test = sol[split:, :]
#y_test = sol_dt[split:, :]

#print("Training set", x_train.shape[0])
#print("Test set", x_test.shape[0])
l = eps*BCL*10

model = Sequential()
#model.add(LSTM(512, implementation=2, input_shape=(l, 42), return_sequences=True))
model.add(Conv1D(1024, 50, padding='causal', kernel_initializer='he_normal', input_shape=(l, 42), activation='relu'))
model.add(Conv1D(1024, 25, padding='causal', kernel_initializer='he_normal', activation='relu'))
model.add(Conv1D(1024, 10, padding='causal', kernel_initializer='he_normal', activation='relu'))
model.add(Activation('relu'))
#for i in range(2):
#    model.add(LSTM(512, implementation=2, return_sequences=True))
#    model.add(LayerNormalization())

model.add(TimeDistributed(Dense(192)))
model.add(TimeDistributed(LayerNormalization()))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(41, activation='linear')))

#model = make_parallel(model, 3)

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=RMSprop())

xtr = (np.array(sol[i:i+batch_size*l, :]).reshape(batch_size, l, -1) for i in range(0, split, batch_size*l))
ytr = (np.array(sol_dt[i:i+batch_size*l, :]).reshape(batch_size, l, -1) for i in range(0, split, batch_size*l))
history = model.fit_generator(zip(xtr, ytr),
                    epochs=epochs, steps_per_epoch=split//(batch_size*l),
                    verbose=1) #, callbacks=[checkpointer])
#                    validation_data=(x_test, y_test))

xte = (sol[i:i+batch_size*l, :].reshape(batch_size, l, -1) for i in range(split, sol.shape[0], batch_size*l))
yte = (sol_dt[i:i+batch_size*l, :].reshape(batch_size, l, -1) for i in range(split, sol.shape[0], batch_size*l))
score = model.evaluate_generator(zip(xte, yte), (sol.shape[0]-split)//(batch_size*l), verbose=0)

