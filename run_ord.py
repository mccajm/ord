import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import ode, odeint

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

y0 = np.array(y0, dtype=np.float32)
BCL = 25 

print("Solving")
st = time.time()
ord_f = ORd(state=y0, pstim=1)
sol = odeint(ord_f.step, y0, np.arange(BCL*250))
print(time.time() - st)

plt.plot(np.arange(BCL*250), sol[:, 0])
plt.show()
