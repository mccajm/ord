import numpy as np
import keras
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.optimizers import Adam
import zarr
from zarr import blosc
blosc.set_nthreads(18)

from selu import dropout_selu
#from uniformization import LayerNormalization
from multi_gpu import make_parallel


batch_size = 32 
epochs = 3 

BCL = 40
eps = 50
pop_size = 100000

print("Loading data")
sol = zarr.open_array('/mnt/ORD_BCL_50_eps_100_1000000.zarr', mode='r', shape=(pop_size*eps*BCL*10, 42), chunks=(1024, None), dtype='float32', fill_value=0.)
sol_dt = zarr.open_array('/mnt/ORD_dt_BCL_50_eps_1000000.zarr', mode='r', shape=(pop_size*eps*BCL*10, 41), chunks=(1024, None), dtype='float32', fill_value=0.)

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

model = Sequential()
with tf.device('/gpu:0'):
    model.add(Dense(2048, kernel_initializer='glorot_uniform', input_shape=(42,)))
    model.add(Lambda(lambda x: dropout_selu(x, 0.5)))

for i in range(3):
    with tf.device(f'/gpu:{i+1}'):
        model.add(Dense(2048, kernel_initializer='he_uniform'))
        model.add(Lambda(lambda x: dropout_selu(x, 0.5)))

model.add(Dense(41, activation='linear'))

#model = make_parallel(model, 4)

model.summary()
model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=0.01))

xtr = (sol[i:i+batch_size] for i in range(0, split, batch_size))
ytr = (sol_dt[i:i+batch_size] for i in range(0, split, batch_size))
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1)  #, save_best_only=True)
history = model.fit_generator(zip(xtr, ytr),
                    epochs=epochs, steps_per_epoch=split//batch_size,
                    verbose=1) #, callbacks=[checkpointer])
#                    validation_data=(x_test, y_test))

xte = (sol[i:i+batch_size] for i in range(split, sol.shape[0], batch_size))
yte = (sol_dt[i:i+batch_size] for i in range(split, sol.shape[0], batch_size))
#x_test = sol[split:, :]
#y_test = sol_dt[split:, :]
score = model.evaluate_generator(zip(xte, yte), (sol.shape[0]-split)//batch_size, verbose=0)
