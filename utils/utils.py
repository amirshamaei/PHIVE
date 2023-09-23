import time
import pandas as pd
import torch
import math
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import numpy.fft as fft
from . import Jmrui

fontsize = 16
def relu(self, x):
    return np.maximum(0, x)

# def simulation(self):
#     mm_signals=0
#     sd = self.parameters["sd"]
#     d = np.random.normal(self.parameters["mmd_mean"][0], sd * self.parameters["mmd_std"][0],
#                          (self.numOfSample, 1))
#     for idx, (F, D, A) in enumerate(zip(self.MM_f, self.MM_d, self.MM_a)):
#         f = np.random.normal(self.parameters["mmf_mean"][idx],sd*self.parameters["mmf_std"][idx],(self.numOfSample,1))
#         a = self.relu(np.random.normal(self.parameters["mma_mean"][idx], sd*self.parameters["mma_std"][idx], (self.numOfSample,1)))
#         mm_signals += self.MM_model(a,f,d,0,A,self.trnfreq * F,D)
#         # print("min:{}, mean:{}, max:{}".format(np.min(f + self.trnfreq *F),np.mean(f + self.trnfreq *F),np.max(f + self.trnfreq *F)))
#
#
#
#     # for idx, (F, D, A) in enumerate(zip(self.parameters["MM_f"], self.parameters["MM_d"], self.parameters["MM_d"])):
#     #     ampl = np.random.normal(self.parameters["met_mean"],sd*self.parameters["met_std"],(self.numOfSample,1))
#
#     ampl = np.asarray(self.parameters["met_mean"]) + np.multiply(np.random.normal(0, 1, size=(self.numOfSample, self.numOfSig)), np.asarray(self.parameters["met_std"])*sd)
#     ampl = self.relu(ampl)
#     shift = np.random.normal(self.parameters["met_shift"][0],sd*self.parameters["met_shift"][1],(self.numOfSample,1))
#     freq = -2 * math.pi * (shift) * self.t.T
#     alpha = np.random.normal(self.parameters["met_damp"][0],sd*self.parameters["met_damp"][1],(self.numOfSample,1))
#     ph = np.random.normal(self.parameters["met_ph"][0],sd*self.parameters["met_ph"][1],(self.numOfSample))
#     signal = np.matmul(ampl[:, 0:self.numOfSig], self.basisset[0:self.sigLen, :].T)
#     # noise = np.random.normal(0, noiseLevel, (ns, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (ns, self.sigLen))
#     # signal = signal + noise
#
#     y = np.multiply(signal, np.exp(-alpha * self.t.T))
#     y = np.multiply(y, np.exp(freq * 1j))
#     y += mm_signals
#
#     y = y.T * np.exp(ph * 1j)
#
#
#     noiseLevel = np.max(y) * self.parameters["noise_level"]
#     noise = np.random.normal(0, noiseLevel, (self.numOfSample, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (self.numOfSample, self.sigLen))
#     y = y + noise.T
#     y_f = np.fft.fftshift(np.fft.fft(y[:,0:1000], axis=0), axes=0)
#     self.plotsppm(y_f, 0, 5, True)
#     plt.title("mean = {} and sd = {}".format(np.mean(self.cal_snrf(y_f)),np.std(self.cal_snrf(y_f))))
#     self.savefig("simulated_signals")
#     Path(self.sim_dir +self.parameters["child_root"]).mkdir(parents=True, exist_ok=True)
#     np.savez(self.sim_dir +self.parameters["child_root"]+ "big_gaba_size_{}_sd_{}".format(self.numOfSample,sd), y , mm_signals.T, ampl, shift, alpha, ph)
#     # np.save(self.sim_dir + "big_gaba_size_{}_sd_{}".format(self.numOfSample, sd), y)
#
# def get_augment(self, signals, n, f_band, ph_band, ampl_band, d_band, noiseLevel):
#     l = []
#     l.append(signals)
#     lens = np.shape(signals)[1]
#     shift_t = f_band * np.random.rand(n * lens) - (f_band / 2)
#     ph_t = ph_band * np.random.rand(n * lens) * math.pi - ((ph_band / 2) * math.pi)
#     ampl_t = 1 + ((ampl_band * np.random.rand(n * lens))-ampl_band/2)
#     d_t = d_band * np.random.rand(n * lens)
#     for i in range(0, lens):
#         signal = np.expand_dims(signals[:, i], 1)
#         numOfSamplei = n
#         freq = -2 * math.pi * (shift_t[i * numOfSamplei:(i + 1) * numOfSamplei]) * self.t
#         ph = ph_t[i * numOfSamplei:(i + 1) * numOfSamplei]
#         ampl = ampl_t[i * numOfSamplei:(i + 1) * numOfSamplei]
#         d = d_t[i * numOfSamplei:(i + 1) * numOfSamplei]
#         y = ampl * signal
#         y = np.multiply(y * np.exp(ph * 1j), np.exp(freq * 1j))
#         y = np.multiply(y, np.exp(-d * self.t))
#         noise = np.random.normal(0, noiseLevel, (len(signal), numOfSamplei)) + 1j * np.random.normal(0, noiseLevel,
#                                                                                                      (len(signal),
#                                                                                                       numOfSamplei))
#         y_i = y + noise
#         l.append(y_i)
#     y = np.hstack(l)
#     return y, ampl_t, d_t, shift_t, ph_t

def loadModel(autoencoder, path):
    # m = LitAutoEncoder(t,signal_norm)
    return autoencoder.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

def cal_snr(self,data, endpoints=128,offset=0):
    return np.abs(data[0, :]) / np.std(data.real[-(offset + endpoints):-(offset+1), :], axis=0)

def  savefig(self, path, plt_tight=True):
    # plt.ioff()
    if plt_tight:
        plt.tight_layout()
    if self.save:
        plt.savefig(self.saving_dir + path + ".svg", format="svg")
        # plt.savefig(self.saving_dir + path + " .png", format="png", dpi=1200)
    plt.clf()

# %%
def tic(self):
    self.start_time = time.time()
    return self
def toc(self,name):
    elapsed_time = (time.time() - self.start_time)
    print("--- %s seconds ---" % elapsed_time)
    timingtxt = open(self.saving_dir + name + ".txt", 'w')
    timingtxt.write(name)
    timingtxt.write("--- %s ----" % elapsed_time)
    timingtxt.close()
    return elapsed_time

# %%


def cal_snrf(self,data_f,range=[2,4],endpoints=128,offset=0):
    p1 = int(ppm2p(self, range[0], data_f.shape[1]))
    p2 = int(ppm2p(self, range[1], data_f.shape[1]))
    return np.max(np.abs(data_f[:,p2:p1]), 1) / (np.std(data_f.real[:,offset:endpoints+offset],axis=1))

def ppm2p(self, r, len):
    r = 4.7 - r
    return int(((self.trnfreq * r) / (1 / (self.t_step * len))) + len / 2)

def ppm2f(self, r):
    return r * self.trnfreq

def zero_fill(self,arr, dim, N):
    """
    Zero fills the given ndarray with the size of N in the requested dimension.

    Parameters:
    arr (numpy.ndarray): The input ndarray to be zero-filled.
    dim (int): The index of the dimension to be zero-filled.
    N (int): The size of the zero-filled dimension.

    Returns:
    numpy.ndarray: The zero-filled ndarray.
    """

    # Create a new shape tuple with N in the requested dimension
    shape = list(arr.shape)
    shape[dim] = N

    # Create a new zero-filled ndarray with the new shape
    out = np.zeros(shape,dtype=arr.dtype)

    # Copy the original data into the new ndarray
    slices = [slice(None)] * arr.ndim
    slices[dim] = slice(None, arr.shape[dim])
    out[slices] = arr

    return out

def zero_fill_torch(self, arr, dim, N):
    shape = list(arr.shape)
    shape[dim] = N

    # Create a new zero-filled ndarray with the new shape
    out = torch.zeros(shape,dtype=arr.dtype).to(arr.device)

    # Copy the original data into the new ndarray
    slices = [slice(None)] * arr.ndim
    slices[dim] = slice(None, arr.shape[dim])
    out[slices] = arr

    return out

def fillppm(self, y1, y2, ppm1, ppm2, rev, alpha=.1, color='red',ax=None):
    # p1 = int(self.ppm2p(ppm1, len(y1)))
    # p2 = int(self.ppm2p(ppm2, len(y1)))
    # n = p2 - p1
    n = len(y1)
    x = np.linspace((ppm1), (ppm2), abs(n))
    if ax ==None:
        plt.fill_between(np.flip(x), y1[:, 0].real,
                         y2[:, 0].real, alpha=alpha, color=color)
        if rev:
            plt.gca().invert_xaxis()
    else:
        ax.fill_between(np.flip(x), y1[:, 0].real,
                         y2[:, 0].real, alpha=alpha, color=color)
        if rev:
            ax.invert_xaxis()

def plotsppm(self, sig, ppm1, ppm2, rev, linewidth=0.3, linestyle='-', color=None):
    p1 = int(self.ppm2p(ppm1, len(sig)))
    p2 = int(self.ppm2p(ppm2, len(sig)))
    n = p2 - p1
    x = np.linspace(int(ppm1), int(ppm2), abs(n))
    sig = np.squeeze(sig)
    df = pd.DataFrame(sig[p2:p1, :].real)
    df['Frequency(ppm)'] = np.flip(x)
    df_m = df.melt('Frequency(ppm)',value_name='Real Signal (a.u.)')
    g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df_m, linewidth=linewidth, linestyle=linestyle,ci="sd")
    plt.legend([],[],frameon=False)
    # g = plt.plot(np.flip(x), sig[p2:p1, :].real, linewidth=linewidth, linestyle=linestyle, color=color)
    plt.tick_params(axis='both', labelsize=fontsize)
    if rev:
        plt.gca().invert_xaxis()

def normalize(inp):
    return (np.abs(inp) / np.abs(inp).max(axis=0)) * np.exp(np.angle(inp) * 1j)

def plotppm(self, sig, ppm1, ppm2, rev, linewidth=0.3, linestyle='-',label=None, mode='real',ax=None):
    # p1 = int(self.ppm2p(ppm1, len(sig)))
    # p2 = int(self.ppm2p(ppm2, len(sig)))
    # n = p2 - p1
    n = len(sig)
    x = np.linspace(ppm1, ppm2, abs(n))
    sig = np.squeeze(sig)
    if mode=='real':
        df = pd.DataFrame({'Real Signal (a.u.)': sig.real})
    if mode=='abs':
        df = pd.DataFrame({'Real Signal (a.u.)': np.abs(sig)})
    df['Frequency(ppm)'] = np.flip(x)
    g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df, linewidth=linewidth, linestyle=linestyle,label=label,ax=ax)
    plt.tick_params(axis='both', labelsize=fontsize)
    if rev:
        if ax == None:
            plt.gca().invert_xaxis()
        else:
            ax.invert_xaxis()
    return g
    # gca = plt.plot(x,sig[p2:p1,0],linewidth=linewidth, linestyle=linestyle)

def plot_basis2(self, basisset, ampl):
    p1 = int(ppm2p(self,4, len(basisset)))
    p2 = int(ppm2p(self,1, len(basisset)))
    for i in range(0, len(basisset.T) - 1):
        plotppm(self,+100* i + fft.fftshift(fft.fft(ampl * self.basisset[:, i]))[p1:p2], 1, 4, False,label=self.met_name[i])
    plotppm(self,100 * (i + 1) + fft.fftshift(fft.fft(self.basisset[:, i + 1]))[p1:p2], 1, 4, True,label=self.met_name[i])
    # plt.legend(self.met_name)
    savefig(self,"Basis" + str(ampl),plt_tight=True)
    plt.tick_params(labelsize=fontsize)


def plot_basis(self, ampl, fr, damp, ph, rng=[1,5]):
    reve = False
    p1 = int(ppm2p(self,rng[0], len(self.basisset)))
    p2 = int(ppm2p(self,rng[1], len(self.basisset)))
    for i in range(0, len(self.basisset.T)):
        vv=fft.fftshift(fft.fft(ampl[0, i] * self.basisset[:len(self.t), i]*np.exp(-2 * np.pi *1j* fr * self.t.T)*np.exp(-1*damp*self.t.T)))
        if i ==len(self.basisset.T)-1:
            reve= True
        ax = plotppm(self,-4 * (i+2) + vv.T[p2:p1], rng[0], rng[1], reve)
        sns.despine(left=True,right=True,top=True)
        plt.text(.1, -4 * (i+2), self.met_name[i],fontsize=8)
        ax.tick_params(left=False)
        ax.set(yticklabels=[])
    plt.tick_params(labelsize=fontsize)


def Lornz(self, ampl, f, d, ph ,Cra, Crfr, Crd):
    return (Cra*ampl) * np.multiply(np.multiply(np.exp(ph * 1j),
                                                np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                 np.exp(-1*(d + Crd) * self.t.T))
def Gauss(self, ampl, f, d, ph, Cra, Crfr, Crd):
    return (Cra*ampl) * np.multiply(np.multiply(np.exp(ph * 1j),
                                                np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                 np.exp(-1*((d + Crd)**2) * self.t.T * self.t.T))

def bland_altman_plot(self,data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

def modified_bland_altman_plot(self,data1, data2, gt=None,c_map=None, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference
    if gt is not None:
        ax = plt.scatter(gt, diff,cmap='Spectral', *args, **kwargs)
    else:
        ax = plt.scatter(range(0, data1.shape[0]), diff ,cmap='Spectral', *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='-',linewidth=3)
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--',linewidth=2)
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--',linewidth=2)
    sns.despine()
    if c_map != None:
        plt.set_cmap(c_map)
        cb = plt.colorbar()
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)

    return ax

def calib_plot(self,ampl_t,y_out, yerr,cmap):
    if cmap==None :
        ax = plt.scatter(x=ampl_t, y=y_out)
    else:
        ax = plt.scatter(x=ampl_t, y=y_out, c=yerr, cmap='Spectral')
        plt.set_cmap(cmap)
        cb = plt.colorbar()
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fontsize)
    ax.axes.spines['right'].set_visible(False)
    ax.axes.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.axes.yaxis.set_ticks_position('left')
    ax.axes.xaxis.set_ticks_position('bottom')
    x0, x1 = ax.axes.get_xlim()
    y0, y1 = ax.axes.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    ax.axes.plot(lims, lims, '--k')
    ax.axes.set_xlabel("True")
    ax.axes.set_ylabel('Predicted')
    plt.tick_params(axis='both', labelsize=fontsize)

def err_plot(self, x , y, yerr, name, cmap):
    if cmap==None :
        ax = plt.scatter(x, y)
    else:
        ax = plt.scatter(x, y, c=yerr, cmap='Spectral')
        cb = plt.colorbar()
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fontsize)
    plt.set_cmap(cmap)
    ax.axes.spines['right'].set_visible(False)
    ax.axes.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.axes.yaxis.set_ticks_position('left')
    ax.axes.xaxis.set_ticks_position('bottom')
    ax.axes.set_xlabel(name)
    ax.axes.set_ylabel('Prediction Error')
    plt.tick_params(axis='both', labelsize=fontsize)

def sigmoid(self,x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig

def wighted_var(self, x, w, **kwargs):
    w_mean = (np.average(x, 0, weights=w,**kwargs))
    w_var = np.average((x - w_mean) ** 2, 0, weights=w,**kwargs)
    return w_var


def plot_MM(self):
    if 'param' in self.MM_type:
        if self.MM:
            mm = 0
            for idx in range(0, self.numOfMM):
                if self.MM_conj == True:
                    x = np.conj(self.MM_model(self.MM_a[idx], 0, 0, 0, self.ppm2f(self.MM_f[idx]), self.MM_d[idx]))
                else:
                    x = (self.MM_model(self.MM_a[idx], 0, 0, 0, self.ppm2f(self.MM_f[idx]), self.MM_d[idx]))
                mm += x
                if idx == self.numOfMM - 1:
                    self.plotppm(-10 * idx + fft.fftshift(fft.fft(x)).T, 0, 5, True)
                else:
                    self.plotppm(-10 * idx + fft.fftshift(fft.fft(x)).T, 0, 5, False)
            self.savefig("MM")
            self.mm = mm.T
            Jmrui.write(Jmrui.makeHeader("tesDB", np.size(self.mm, 0), np.size(self.mm, 1), 0.25, 0, 0,
                                         1.2322E8), self.mm, self.saving_dir + '_mm.txt')

    plot_basis2(self,self.basisset, 2)