import time

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

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
y0 = np.tile(np.array(y0).astype(np.float64).reshape(-1, 1), pop_size)
BCL = 25

print("Solving")
ord_f = ORd(y0=y0, pstim=1, CL=BCL)
eps = 50
sol = np.empty((BCL*eps, pop_size), dtype=np.float64)
cur_state = y0
sol[0, :] = cur_state[0, :]
ts = np.linspace(0, 50, num=BCL*eps, dtype=np.float64)
st = time.time()
for i, t in enumerate(tqdm(ts)):
    if i == 0:
        pass

    dt = ord_f.step(cur_state, t)
    cur_state += dt[0, :, :] / eps
    sol[i, :] = cur_state[0, :]

print(time.time() - st)

plt.plot(np.arange(BCL*eps), sol[:, 0])
plt.show()
