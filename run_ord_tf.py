import time

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
import zarr
from zarr import blosc
blosc.set_nthreads(18)
from tqdm import tqdm

from ord_endo_tf import ORd


# Initial Conditions
v, nai, nass, ki, kass = -87, 7, 7, 150, 150 
cai, cass, cansr, cajsr, m = 1.0e-4, 1.0e-4, 1.2, 1.2, 0
hf, hs, j, hsp, jp = 1, 1, 1, 1, 1
mL, hL, hLp, a, iF = 0, 1, 1, 0, 1
iS, ap, iFp, iSp, d = 1, 0, 1, 1, 0
ff, fs, fcaf, fcas, jca = 1, 1, 1, 1, 1
nca, ffp, fcafp, xrf, xrs = 0, 1, 1, 0, 0
xs1, xs2, xk1, Jrelnp, Jrelp = 0, 0, 1, 0, 0
CaMKt = 0
    
y0 = [v, nai, nass, ki,  kass,
      cai, cass, cansr, cajsr, m,
      hf, hs, j, hsp, jp,
      mL, hL, hLp, a, iF,
      iS, ap, iFp, iSp, d,
      ff, fs, fcaf, fcas, jca,
      nca, ffp, fcafp, xrf, xrs,
      xs1, xs2, xk1, Jrelnp,  Jrelp, CaMKt]

# Number of models to train
pop_size = 100000
y0 = np.tile(np.array(y0).astype(np.float32).reshape(-1, 1), pop_size)
BCL = 40

print("Solving")
with tf.Session() as sess:
    ord_f = ORd(y0=y0, pstim=1, sess=sess, CL=BCL)
    def step(state, t):
        dstates = sess.run([ord_f.dstates], feed_dict={ord_f.t: np.array(t), ord_f.state: state.reshape(-1, 1)})
        return dstates[0].flatten()

    eps = 50
    sol = zarr.open_array('/mnt/ORD_BCL_50_eps_100_1000000.zarr', mode='w', shape=(pop_size*eps*BCL*10, 42), chunks=(1024, None), dtype='float32', fill_value=0.)
    sol_dt = zarr.open_array('/mnt/ORD_dt_BCL_50_eps_1000000.zarr', mode='w', shape=(pop_size*eps*BCL*10, 41), chunks=(1024, None), dtype='float32', fill_value=0.)
    cur_state = y0
    ts = np.linspace(0, BCL*10, num=BCL*10*eps, dtype=np.float32)
    st = time.time()
    tq = tqdm(ts)
    k = 0
    for i, t in enumerate(tq):
        #if i % eps*BCL > 300 and i % 10 != 0:  # We only need high accuracy when near BCL
        #    continue

        sol[k*pop_size:k*pop_size+pop_size, :-1] = cur_state.swapaxes(0, 1)
        dt, Istim = sess.run([ord_f.dstates, ord_f.Istim], feed_dict={ord_f.t: np.array(t), ord_f.state: cur_state})
        cur_state += dt / eps
        sol[k*pop_size:k*pop_size+pop_size, -1] = Istim.reshape(-1)

        sol_dt[k*pop_size:k*pop_size+pop_size, :] = dt.swapaxes(0, 1)
        tq.set_postfix({'dv': dt[0, 0]})
        k += 1

    print(time.time() - st)

#plt.plot(np.arange(BCL*eps), sol[:, 0])
#plt.show()
