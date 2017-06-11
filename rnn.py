import numpy as np
import keras
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.optimizers import Adam, RMSprop
import zarr
from zarr import blosc
blosc.set_nthreads(18)

from normalization import LayerNormalization
from multi_gpu import make_parallel


batch_size = 8192 
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
x_train = sol[:split, :]
y_train = sol_dt[:split, :]
x_test = sol[split:, :]
y_test = sol_dt[split:, :]

print("Training set", x_train.shape[0])
print("Test set", x_test.shape[0])

model = Sequential()
model.add(GRU(128, implementation=2, unroll=True, input_shape=(42,)))
model.add(LayerNormalization())
for _ in range(4):
    model.add(GRU(128, implementation=2, unroll=True))
    model.add(LayerNormalization())

model.add(Dense(41, activation='linear'))

#model = make_parallel(model, 3)

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=RMSprop())

#checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1)  #, save_best_only=True)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1) #, callbacks=[checkpointer])
#                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

