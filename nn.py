import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from multi_gpu import make_parallel


batch_size = 128
epochs = 20

BCL = 20
eps = 100
pop_size = 2000000

print("Loading data") 
sol = np.memmap(f"/mnt/ORd_endo_bcl_{BCL}_state.bin", shape=(BCL*eps*pop_size, 41), dtype=np.float32, mode='r')
sol_dt = np.memmap(f"/mnt/ORd_endo_bcl_{BCL}_dt.bin", shape=(BCL*eps*pop_size, 41), dtype=np.float32, mode='r')

print("Splitting sets")
split = int(sol.shape[0] * 0.8)
x_train = sol[:split, :]
y_train = sol_dt[:split, :]
x_test = sol[split:, :]
y_test = sol_dt[split:, :]

print("Training set", x_train.shape[0])
print("Test set", x_test.shape[0])

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(41,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(41, activation='linear'))

model = make_parallel(model, range(2))

model.summary()

model.compile(loss='mean_squared_error',
              optimizer=Adam())

import ipdb
ipdb.set_trace()
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

import ipdb
ipdb.set_trace()
#score = model.evaluate(x_test, y_test, verbose=0)

