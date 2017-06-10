import time

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.integrate import odeint

from ord_endo import ORd


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
pop_size = 1000000
y0 = np.tile(np.array(y0).astype(np.float32).reshape(-1, 1), pop_size)
BCL = 25

print("Solving")
with tf.Session() as sess:
    ord_f = ORd(y0=y0, pstim=1, sess=sess, CL=BCL)
    def step(state, t):
        dstates = sess.run([ord_f.dstates], feed_dict={ord_f.t: np.array(t), ord_f.state: state.reshape(-1, 1)})
        return dstates[0].flatten()

    eps = 50
    sol = np.empty((BCL*eps, pop_size), dtype=np.float32)
    cur_state = y0
    sol[0, :] = cur_state[0, :]
    ts = np.linspace(0, 50, num=BCL*eps, dtype=np.float32)
    st = time.time()
    for i, t in enumerate(tqdm(ts)):
        if i == 0:
            pass

        dt = sess.run([ord_f.dstates], feed_dict={ord_f.t: np.array(t), ord_f.state: cur_state})[0]
        cur_state += dt / eps
        sol[i, :] = cur_state[0, :]

    print(time.time() - st)

plt.plot(np.arange(BCL*eps), sol[:, 0])
plt.show()
