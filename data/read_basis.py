import numpy as np
import scipy.io as sio

basis_set = sio.loadmat('basis_set.mat').get('basis_set')
names_trimmed = ["Cr", "GPC", "Gln", "Ins", "MM_mea", "NAAG", "NAA", "PCr", "Glu"]
names = ["Act","Ala","Glc","Asp","Glc_B","Cho","Cr","GABA","GPC","GSH","Gln","Ins","Lac","MM_mea","NAAG","NAA","PCh","PCr","Scyllo","Tau","TwoHG","Glu","Gly","Lip_c","mm3","mm4"];

for i, name in enumerate(names_trimmed):
    index = names.index(name)
    metabolite = basis_set[index, :]
    sio.savemat(f'basis/{name}.mat', {f'{name}': metabolite})
