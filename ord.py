"""
O'hara-Rudy Human Ventricular Model (2011)

Original Matlab file from:
http://rudylab.wustl.edu/research/cell/code/AllCodes.html

Related Journal Paper
http://www.ncbi.nlm.nih.gov/pubmed/21637795

Converted to TensorFlow by Adam McCarthy
"""

from math import pi

import numpy as np

from scipy.special import expit as sigmoid


class ORd:

    def __init__(self, nao=140, cao=1.8, ko=5.4, flag_ode=1, pstim=0, CL=1000, **kwargs):
        # State Variables
        for k, v in kwargs:
            setattr(self, k, v)
 
        # Extracellular Ionic Concentrations (mM)
        self.concentrations = {}
        self.concentrations['nao'] = nao
        self.concentrations['cao'] = cao
        self.concentrations['ko'] = ko

        # Physical constrants
        self.R = 8314.0  # J/kmol/k
        self.T = 310.0  # k
        self.F = 96485.0  # C/mol

        # Cell Geometry
        #  (approx by a cylinder length L and radius rad)
        self.L = 0.01  # cm
        self.rad = 0.0011  # cm
        self.vcell = 1000 * pi  # uL

        # Geometric Area
        self.Ageo = 2 * pi * self.rad**2 + 2 * pi * self.rad * L

        # Capacitive Area
        self.Acap = 2 * self.Ageo

        # Compartment Volumes (uL)
        self.vmyo = 0.68 * self.vcell
        self.vnsr = 0.0552 * self.vcell
        self.vjsr = 0.0048 * self.vcell
        self.vss = 0.02 * self.vcell

        # Reversal Potentials
        self.ENa = (self.R * self.T / self.F) * np.log(self.nao / self.nai)
        self.EK = (self.R * self.T / self.F) * np.log(self.ko / self.ki)
        self.PKNa = 0.01833
        self.EKs = (self.R * self.T / self.F) * np.log((self.ko + self.PKNa * self.nao) / (self.ki + self.PKNa * self.nai))

        self.KmCamK = 0.15
        self.aCaMK = 0.05
        self.bCaMK = 0.00068
        self.CaMKo = 0.05
        self.KmCaM = 0.0015
 
    def update_CaMK(self)
        # Update CaMK in state
        self.CaMKb = self.CaMKo * (1 - self.CAMkt) / (1 + self.kmCaM / self.cass)
        self.CaMka = self.CaMkb + self.CaMKt
        self.dCaMKt = self.aCaMK * self.aCaMKb * (self.CaMKb + self.CaMKt) - self.bCAMK * CaMKt

    def calc_INa(self):
        # INa current
        ss = 1.0 / (1.0 + np.exp(( - (self.v + 39.57)) / 9.871))
        tm = 1.0 / (6.765 * np.exp((self.v + 11.64) / 34.77) + 8.552 * np.exp( - (self.v + 77.42) / 5.955))
        hss = 1.0 / (1 + np.exp((self.v + 82.90) / 6.086))
        thf = 1.0 / (1.432e-5 * np.exp( - (self.v + 1.196) / 6.285) + 6.149 * np.exp((self.v + 0.5096) / 20.27))
        ths = 1.0 / (0.009794 * np.exp( - (self.v + 17.95) / 28.05) + 0.3343 * np.exp((self.v + 5.730) / 56.66))
        Ahf = 0.99
        Ahs = 1.0 - Ahf
        h = Ahf * self.hf + Ahs * self.hs
        jss = hss
        tj = 2.038 + 1.0 / (0.02136 * np.exp( - (self.v + 100.6) / 8.281) + 0.3052 * np.exp((v + 0.9941) / 38.45))
        hssp = 1.0 / (1 + np.exp((self.v + 89.1) / 6.086))
        thsp = 3.0 * ths
        hp = Ahf * self.hf + Ahs * self.hsp
        tjp = 1.46 * tj
        dm = (mss - m) / tm
        dhf = (hss - self.hf) / thf
        dhs = (hss - self.hs) / ths
        dj = (jss - self.j) / tj
        dhsp = (hssp - self.hsp) / thsp
        djp = (jss - self.jp) / tjp
        GNa = 75
        fINap = (1.0 / (1.0 + self.KmCaMK / self.CaMKa))
        
        INa = GNa * (self.v - self.ENa) * self.m^3.0 * ((1.0 - fINap) * h * self.j + fINap * hp * self.jp)
        return INa

    def calc_Ito(self):
        ass = 1.0 / (1.0 + np.exp(( - (self.v - 14.34)) / 14.82))
        ta = 1.0515 / (1.0 / (1.2089 * (1.0 + np.exp( - (self.v - 18.4099) / 29.3814))) + ...
                             3.5 / (1.0 + np.exp((self.v + 100.0) / 29.3814)))
        da = (ass - self.a) / ta
        iss = 1.0 / (1.0 + np.exp((self.v + 43.94) / 5.711))
        delta_epi = 1.0
        tiF = 4.562 + 1 / (0.3933 * np.exp(( - (self.v + 100.0)) / 100.0) + 0.08004 * np.exp((self.v + 50.0) / 16.59))
        tiS = 23.62 + 1 / (0.001416 * np.exp(( - (self.v + 96.52)) / 59.05) + 1.780e-8 * np.exp((self.v + 114.1) / 8.079))
        tiF = tiF * delta_epi
        tiS = tiS * delta_epi
        AiF = 1.0 / (1.0 + np.exp((self.v - 213.6) / 151.2))
        AiS = 1.0 - AiF
        diF = (iss - self.iF) / tiF
        diS = (iss - self.iS) / tiS
        i = AiF * self.F + AiS * self.iS
        assp = 1.0 / (1.0 + np.exp(( - (self.v - 24.34)) / 14.82))
        dap = (assp - self.ap) / ta
        dti_develop = 1.354 + 1.0e-4 / (np.exp((self.v - 167.4) / 15.89) + np.exp( - (self.v - 12.23) / 0.2154))
        dti_recover = 1.0 - 0.5 / (1.0 + np.exp((self.v + 70.0) / 20.0))
        tiFp = dti_develop * dti_recover * tiF
        tiSp = dti_develop * dti_recover * tiS
        diFp = (iss - self.iFp) / tiFp
        diSp = (iss - self.iSp) / tiSp
        ip = AiF * self.iFp + AiS * self.iSp
        fItop = (1.0 / (1.0 + self.KmCaMK / self.CaMKa))
        Gto = 0.02

        Ito = Gto * (self.v - self.EK) * ((1.0 - fItop) * self.a * i + fItop * self.ap * ip)
        return Ito

    def calc_ICaL_ICaNa_ICaK(self):
        dss = 1.0 / (1.0 + np.exp(( - (self.v + 3.940)) / 4.230))
        td = 0.6 + 1.0 / (np.exp( - 0.05 * (self.v + 6.0)) + np.exp(0.09 * (self.v + 14.0)))
        fss = 1.0 / (1.0 + np.exp((self.v + 19.58) / 3.696))
        tff = 7.0 + 1.0 / (0.0045 * np.exp( - (self.v + 20.0) / 10.0) + 0.0045 * np.exp((self.v + 20.0) / 10.0))
        tfs = 1000.0 + 1.0 / (0.000035 * np.exp( - (self.v + 5.0) / 4.0) + 0.000035 * np.exp((self.v + 5.0) / 6.0))
        Aff = 0.6
        Afs = 1.0 - Aff
        f = Aff * self.ff + Afs * self.fs
        fcass = fss
        tfcaf = 7.0 + 1.0 / (0.04 * np.exp( - (self.v - 4.0) / 7.0) + 0.04 * np.exp((self.v - 4.0) / 7.0))
        tfcas = 100.0 + 1.0 / (0.00012 * np.exp( - self.v / 3.0) + 0.00012 * np.exp(self.v / 7.0))
        Afcaf = 0.3 + 0.6 / (1.0 + np.exp((self.v - 10.0) / 10.0))
        Afcas = 1.0 - Afcaf
        fca = Afcaf * self.fcaf + Afcas * self.fcas
        tjca = 75.0
        ktaup = 2.5
        tffp = ktaup * tff
        fp = Aff * self.ffp + Afs * self.fs
        tfcafp = ktaup * tfcaf
        fcap = Afcaf * self.fcafp + Afcas * self.fcas
        Kmn = 0.002
        k2n = 1000.0
        km2n = self.jca * 1.0
        anca = 1.0 / (k2n / km2n + (1.0 + Kmn / self.cass)**4.0)
        dnca = (anca * k2n - self.nca * km2n)
        PhiCaL = 4.0 * self.vffrt * (cass * np.exp(2.0 * self.vfrt) - 0.341 * cao) / (np.exp(2.0 * self.vfrt) - 1.0)
        PhiCaNa = 1.0 * self.vffrt * (0.75 * self.nass * np.exp(1.0 * self.vfrt) - 0.75 * self.nao) / (np.exp(1.0 * self.vfrt) - 1.0)
        PhiCaK = 1.0 * self.vffrt * (0.75 * self.kss * np.exp(1.0 * self.vfrt) - 0.75 * self.ko) / (np.exp(1.0 * self.vfrt) - 1.0)
        PCa = 0.0001
        PCap = 1.1 * PCa
        PCaNa = 0.00125 * PCa
        PCaK = 3.574e - 4 * PCa
        PCaNap = 0.00125 * PCap
        PCaKp = 3.574e - 4 * PCap
        dd = (dss - self.d) / td
        dff = (fss - self.ff) / self.tff
        dfs = (fss - self.fs) / self.tfs
        dfcaf = (fcass - self.fcaf) / tfcaf
        dfcas = (fcass - self.fcas) / tfcas
        djca = (fcass - self.jca) / tjca
        dffp = (fss - self.ffp) / tffp
        dfcafp = (fcass - self.fcafp) / tfcafp
        fICaLp = (1.0 / (1.0 + self.KmCaMK / self.CaMKa))

        # peak CaL (from supplementary mat)
        ICaLmax  =  PCa * PhiCaL 

        ICaL = ((1.0 - fICaLp) * PCa * PhiCaL * self.d * (f * (1.0 - self.nca) + self.jca * fca * self.nca) + 
                      fICaLp * PCap * PhiCaL * self.d * (fp * (1.0 - self.nca) + self.jca * fcap * self.nca))
        ICaNa = ((1.0 - fICaLp) * PCaNa * PhiCaNa * self.d * (f * (1.0 - self.nca) + self.jca * fca * self.nca) + 
                           fICaLp * PCaNap * PhiCaNa * self.d * (fp * (1.0 - self.nca) + self.jca * fcap * self.nca))
        ICaK = ((1.0 - fICaLp) * PCaK * PhiCaK * self.d * (f * (1.0 - self.nca) + self.jca * fca * self.nca) + 
                          fICaLp * PCaKp * PhiCaK * self.d * (fp * (1.0 - self.nca) + self.jca * fcap * self.nca))

        return (ICaL, ICaNa, ICaK)

    def calc_IKr(self):
        xrss = 1.0 / (1.0 + np.exp(( - (self.v + 8.337)) / 6.789))
        txrf = 12.98 + 1.0 / (0.3652 * np.exp((self.v - 31.66) / 3.869) +
                    4.123e-5 * np.exp(( - (self.v - 47.78)) / 20.38))
        txrs = 1.865 + 1.0 / (0.06629 * np.exp((self.v - 34.70) / 7.355) +
                    1.128e-5 * np.exp(( - (self.v - 29.74)) / 25.94))
        Axrf = 1.0 / (1.0 + np.exp((self.v + 54.81) / 38.21))
        Axrs = 1.0 - Axrf
        dxrf = (xrss - self.xrf) / txrf
        dxrs = (xrss - self.xrs) / txrs
        xr = Axrf * self.xrf + Axrs * self.xrs
        rkr = 1.0 / (1.0 + np.exp((self.v + 55.0) / 75.0)) * 1.0 / (1.0 + np.exp((self.v - 10.0) / 30.0))
        GKr = 0.046
        
        IKr = GKr * np.sqrt(ko / 5.4) * self.xr * self.rkr * (self.v - self.EK)
        return IKr

    def calc_IKs(self):
        xs1ss = 1.0 / (1.0 + np.exp(( - (self.v + 11.60)) / 8.932))
        txs1 = 817.3 + 1.0 / (2.326e-4 * np.exp((self.v + 48.28) / 17.80) +
                        0.001292 * np.exp(( - (self.v + 210.0)) / 230.0))
        dxs1 = (xs1ss - self.xs1) / txs1
        txs2 = 1.0 / (0.01 * np.exp((self.v - 50.0) / 20.0) + 0.0193 * np.exp(( - (v + 66.54)) / 31.0))
        dxs2 = (xs1ss - self.xs2) / txs2
        KsCa = 1.0 + 0.6 / (1.0 + (3.8e-5 / self.cai)**1.4)
        GKs = 0.0034

        IKs = GKs * self.KsCa * self.xs1 * self.xs2 * (self.v - self.EKs)
        return IKs

    def calc_IK1(self):
        xk1ss = 1.0 / (1.0 + np.exp( - (self.v + 2.5538 * self.ko + 144.59) / (1.5692 * self.ko + 3.8115)))
        txk1 = 122.2 / (np.exp(( - (self.v + 127.2)) / 20.36) + np.exp((self.v + 236.8) / 69.33))
        dxk1 = (xk1ss - self.xk1) / txk1
        rk1 = 1.0 / (1.0 + np.exp((self.v + 105.8 - 2.6 * self.ko) / 9.493))
        GK1 = 0.1908 * 2.3238 * np.sqrt(self.ko / 5.4)

        IK1 = GK1 * rk1 * self.xk1 * (self.v - self.EK)
        return IK1

    def calc_INaCa(self):
        kna1, kna2, kna3, kasymm = 15.0, 5.0, 88.12, 12.5
        wna, wca, wnaca, KmCaAct = 6.0e4, 6.0e4, 5.0e3, 150.0e-6
        kcaon, kcaoff, qna, qca = 1.5e6, 5.0e3, 0.5224, 0.1670
        zna, Gncx, zca = 1.0, 0.0008, 2.0
        hca = np.exp((ca * self.v * self.F) / (self.R * self.T))
        hna = np.exp((qna * self.v * self.F) / (self.R * self.T))

        # INaCa_i current
        h1, h2 = 1 + self.nai / kna3 * (1 + hna), (self.nai * hna) / (kna3 * h1)
        h3, h4 = 1.0 / h1, 1.0 + self.nai / kna1 * (1 + self.nai / kna2)
        h5, h6 = self.nai**2 / (h4 * kna1 * kna2), 1.0 / h4
        h7, h8 = 1.0 + self.nao / kna3 * (1.0 + 1.0 / hna), self.nao / (kna3 * hna * h7)
        h9, h10 = 1.0 / h7, kasymm + 1.0 + self.nao / kna1 * (1.0 + self.nao / kna2)
        h11, h12 = self.nao**2 / (h10 * kna1 * kna2), 1.0 / h10
        
        k1, k2, k3p, k3pp = h12 * self.cao * kcaon, kcaoff, h9 * wca, h8 * wnaca      
        k3, k4p, k4pp = k3p + k3pp, h3 * wca / hca, h2 * wnaca, k4p + k4pp
        k5, k6, k7, k8 = kcaoff, h6 * self.cai * kcaon, h5 * h2 * wna, h8 * h11 * wna
        
        x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3) x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
        x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3) x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
        
        E1, E2 = x1 / (x1 + x2 + x3 + x4), x2 / (x1 + x2 + x3 + x4)
        E3, E4 = x3 / (x1 + x2 + x3 + x4), x4 / (x1 + x2 + x3 + x4)
        
        allo = 1.0 / (1.0 + (self.KmCaAct / self.cai)**2.0)   
        JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp   
        JncxCa = E2 * k2 - E1 * k1
 
        INaCa_i = 0.8 * Gncx * allo * (zna * JncxNa + zca * JncxCa)

        h1, h2 = 1 + self.nass / kna3 * (1 + hna), (nass * hna) / (kna3 * h1)
        h3, h4 = 1.0 / h1, 1.0 + nass / kna1 * (1 + nass / kna2)
        h5, h6 = self.nass**2 / (h4 * kna1 * kna2), 1.0 / h4
        h7, h8 = 1.0 + self.nao / kna3 * (1.0 + 1.0 / hna), self.nao / (kna3 * hna * h7)
        h9, h10 = 1.0 / h7, kasymm + 1.0 + self.nao / kna1 * (1 + self.nao / kna2)
        h11, h12 = self.nao**2 / (h10 * kna1 * kna2), 1.0 / h10
        
        k1, k2, k3, k4 = h12 * self.cao * kcaon, kcaoff, h9 * wca, h8 * wnaca
        k3, k4p, k4pp, k4 = k3p + k3pp, k4p = h3 * wca / hca, h2 * wnaca, k4p + k4pp
        k5, k6, k7, k8 = kcaoff, h6 * self.cass * kcaon, h5 * h2 * wna, h8 * h11 * wna
        1
        x1 = k2 * k4 * (k7 + k6) + k5 * k7 * (k2 + k3)
        x2 = k1 * k7 * (k4 + k5) + k4 * k6 * (k1 + k8)
        x3 = k1 * k3 * (k7 + k6) + k8 * k6 * (k2 + k3)
        x4 = k2 * k8 * (k4 + k5) + k3 * k5 * (k1 + k8)
        
        E1 = x1 / (x1 + x2 + x3 + x4)
        E2 = x2 / (x1 + x2 + x3 + x4)
        E3 = x3 / (x1 + x2 + x3 + x4)
        E4 = x4 / (x1 + x2 + x3 + x4)
        
        allo = 1.0 / (1.0 + (self.KmCaAct / self.cass)**2.0)
        JncxNa = 3.0 * (E4 * k7 - E1 * k8) + E3 * k4pp - E2 * k3pp
        JncxCa = E2 * k2 - E1 * k1

        INaCa_ss = 0.2 * Gncx * allo * (zna * JncxNa + zca * JncxCa)
        
        INaCa = INaCa_i + INaCa_ss
        return INaCa
                         
    def step(self):
        self.vffrt = self.v * self.F**2 / (self.R * self.T)
        self.vfrt = self.v * self.F / (self.R * self.T)

        INa = self.calc_INa()
        Ito = self.calc_Ito()
        ICaL, ICaNa, ICaK = self.calc_ICaL_ICaNa_ICaK()
        IKr = self.calc_IKr()
        IKs = self.calc_IKs()
        IK1 = self.calc_IK1()
        INaCa = self.INaCa()

