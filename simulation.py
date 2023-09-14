import json
import math

import numpy as np
json_file_path = 'runs/exp3.json'
with open(json_file_path, 'r') as j:
    contents = json.loads(j.read())[0]

self.sigLen = contents['sigLen']
self.truncSigLen = contents['truncSigLen']
self.BW = 1 / self.t_step
self.f = np.linspace(-self.BW / 2, self.BW / 2, self.sigLen)
self.t = np.arange(0, self.sigLen) * self.t_step
self.t = np.expand_dims(self.t, 1)

def Lornz( ampl, f, d, ph, Crfr, Crd):
    return ampl * np.multiply(np.multiply(np.exp(ph * 1j),
                                                np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                 np.exp(-1*(d + Crd) * self.t.T))
def Gauss(ampl, f, d, ph, Crfr, Crd):
    return ampl * np.multiply(np.multiply(np.exp(ph * 1j),
                                                np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                 np.exp(-1*((d + Crd)**2) * self.t.T * self.t.T))

def Gauss(self, ampl, f, d, ph, Crfr, Cra, Crd):
    return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                   torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                              torch.exp(-(d+Crd)**2 * self.t.T*self.t.T))

for f,d,a in zip(contents["MM_f"],contents["MM_d"],contents["MM_d"]):
    f = np.random.normal()