import csv
import gc
import time
import nibabel as nib
from pathlib import Path
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import mat73
import pandas as pd
import scipy
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.linear_model import LinearRegression
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch
import math
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.io as sio
import numpy.fft as fft
from torchsummary import summary
from utils import Jmrui, watrem
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from Model5 import Encoder_Model
fontsize = 16


class Engine():
    def __init__(self, parameters):
        if parameters['intr_plot'] == False:
            plt.ioff()
        else:
            plt.ion()
        self.parameters = parameters
        self.saving_dir = parameters['parent_root'] + parameters['child_root'] + parameters['version']
        self.epoch_dir =  "epoch/"
        self.loging_dir = parameters['parent_root'] + parameters['child_root']
        self.data_dir = parameters['data_dir']
        self.data_dir_ny = parameters['data_dir_ny']
        self.basis_dir = parameters['basis_dir']
        self.test_data_root = parameters['test_data_root']
        Path(self.saving_dir).mkdir(parents=True, exist_ok=True)
        Path(self.saving_dir+self.epoch_dir).mkdir(parents=True, exist_ok=True)
        self.max_epoch = parameters['max_epoch']
        self.batchsize = parameters['batchsize']
        self.numOfSample = parameters['numOfSample'];
        self.t_step = parameters['t_step']
        self.trnfreq = parameters['trnfreq']
        self.nauis = parameters['nauis']
        self.save = parameters['save']
        self.tr = parameters['tr']
        self.betas = parameters['betas']
        self.depths = parameters['depths']
        self.ens = parameters['ens']
        self.met_name = parameters['met_name']
        self.BW = 1 / self.t_step
        self.f = np.linspace(-self.BW / 2, self.BW / 2, 4096)
        try:
            basis_name = parameters["basis_name"]
        except:
            basis_name = "data"

        try:
            self.num_of_workers = parameters["num_of_workers"]
        except:
            self.num_of_workers = 0
        if self.basis_dir is not None:
            self.basisset = (sio.loadmat(self.basis_dir).get(basis_name)).T
            if parameters['basis_conj']:
                self.basisset = np.conj(self.basisset)
            try:
                if parameters['norm_basis']:
                    pass
                    # self.basisset = self.normalize(self.basisset)
            except:
                pass
        self.wr = parameters['wr']
        self.data_name = parameters['data_name']
        if self.data_dir is not None:
            try:
                self.dataset = scipy.io.loadmat(self.data_dir).get(self.data_name).T
            except:
                try:
                    self.dataset = mat73.loadmat(self.data_dir).get(self.data_name).T
                except:
                    self.dataset = np.load(self.data_dir).T

        self.numOfSig = parameters['numOfSig']
        self.sigLen = parameters['sigLen']
        self.truncSigLen = parameters['truncSigLen']
        self.org_truncSigLen = self.truncSigLen
        self.BW = 1 / self.t_step
        self.f = np.linspace(-self.BW / 2, self.BW / 2, self.sigLen)
        self.t = np.arange(0, self.sigLen) * self.t_step
        self.t = np.expand_dims(self.t, 1)
        self.MM = parameters['MM']
        self.MM_f = parameters['MM_f']
        self.MM_d = np.array(parameters['MM_d'])
        self.MM_a = parameters['MM_a']
        self.MM_plot = parameters['MM_plot']
        self.pre_plot = parameters['pre_plot']
        self.basis_need_shift = parameters['basis_need_shift']
        self.basisset = self.normalize(self.basisset)
        self.aug_params = parameters['aug_params']
        self.tr_prc = parameters['tr_prc']
        self.in_shape= parameters['in_shape']
        self.enc_type = parameters['enc_type']
        self.banorm = parameters['banorm']
        if self.basis_dir and parameters['max_c'] is not None:
            max_c = np.array(parameters['max_c'])
            min_c = np.array(parameters['min_c'])
            self.min_c = (min_c) / np.max((max_c));
            self.max_c = (max_c) / np.max((max_c));
        self.reg_wei = parameters['reg_wei']
        self.data_conj = parameters['data_conj']
        self.test_nos = parameters['test_nos']
        self.quality_filt = parameters['quality_filt']
        self.test_name = parameters['test_name']
        self.beta_step = parameters['beta_step']
        self.MM_type = parameters['MM_type']
        self.MM_dir = parameters['MM_dir']
        self.MM_constr = parameters['MM_constr']
        self.comp_freq = parameters['comp_freq']
        if self.MM_dir is not None:
            self.mm = sio.loadmat(self.MM_dir).get("data")
            self.mm[0] = self.mm[0] - 1*fft.fftshift(fft.fft(self.mm, axis=0))[0]
        self.sim_params = parameters['sim_params']
        if self.sim_params is not None:
            for i, val in enumerate(self.sim_params):
                if isinstance(val,str):
                    self.sim_params[i] = getattr(self, self.sim_params[i])
        try:
            self.test_params = parameters['test_params']

            if self.test_params is not None:
                for i, val in enumerate(self.test_params):
                    if isinstance(val,str):
                        self.test_params[i] = getattr(self, self.test_params[i])
        except:
            pass

        if self.MM:
            if parameters['MM_model'] == "lorntz":
                self.MM_model = self.Lornz
                self.MM_d = (np.pi * self.MM_d)
                # self.MM_d = (np.pi ** self.MM_d) * ((self.MM_d) ** 2) / (2 * np.log(2))
            if parameters['MM_model'] == "gauss":
                self.MM_model = self.Gauss
                self.MM_d = self.MM_d * self.trnfreq
                self.MM_d = (np.pi * self.MM_d)/(2*np.sqrt(np.log(2)))
            self.numOfMM = len(self.MM_f)
            if self.MM_type == 'single' or self.MM_type == 'single_param':
                self.met_name.append("MM")
        else:
            self.numOfMM = 0
        self.heatmap_cmap = sns.diverging_palette(20, 220, n=200)
        self.sim_now = parameters['sim_order'][0]
        self.sim_dir = parameters['sim_order'][1]
        try:
            self.kw = self.parameters['kw']
        except:
            self.parameters['kw'] = 3
            self.kw = 3
        try:
            self.MM_fd_constr = self.parameters['MM_fd_constr']
        except:
            self.MM_fd_constr = True

        try:
            self.MM_conj = self.parameters['MM_conj']
        except:
            self.MM_conj = True
            print("take care MM is conjucated!")

        if self.MM_conj == False:
            self.MM_f = [zz - 4.7 for zz in  self.MM_f]

        if self.basis_need_shift[0] == True:
            self.basisset = self.basisset[:, :] * np.exp(
                2 * np.pi * self.ppm2f(self.basis_need_shift[1]) * 1j * self.t)
        if self.parameters['domain'] == 'freq':
            self.p1 = int(self.ppm2p(self.parameters['fbound'][2], (self.truncSigLen)))
            self.p2 = int(self.ppm2p(self.parameters['fbound'][1], (self.truncSigLen)))
            self.in_size = int(self.p2-self.p1)
            # %%

    def relu(self,x):
        return np.maximum(0,x)
    def simulation(self):
        mm_signals=0
        sd = self.parameters["sd"]
        d = np.random.normal(self.parameters["mmd_mean"][0], sd * self.parameters["mmd_std"][0],
                             (self.numOfSample, 1))
        for idx, (F, D, A) in enumerate(zip(self.MM_f, self.MM_d, self.MM_a)):
            f = np.random.normal(self.parameters["mmf_mean"][idx],sd*self.parameters["mmf_std"][idx],(self.numOfSample,1))
            a = self.relu(np.random.normal(self.parameters["mma_mean"][idx], sd*self.parameters["mma_std"][idx], (self.numOfSample,1)))
            mm_signals += self.MM_model(a,f,d,0,A,self.trnfreq * F,D)
            # print("min:{}, mean:{}, max:{}".format(np.min(f + self.trnfreq *F),np.mean(f + self.trnfreq *F),np.max(f + self.trnfreq *F)))



        # for idx, (F, D, A) in enumerate(zip(self.parameters["MM_f"], self.parameters["MM_d"], self.parameters["MM_d"])):
        #     ampl = np.random.normal(self.parameters["met_mean"],sd*self.parameters["met_std"],(self.numOfSample,1))

        ampl = np.asarray(self.parameters["met_mean"]) + np.multiply(np.random.normal(0, 1, size=(self.numOfSample, self.numOfSig)), np.asarray(self.parameters["met_std"])*sd)
        ampl = self.relu(ampl)
        shift = np.random.normal(self.parameters["met_shift"][0],sd*self.parameters["met_shift"][1],(self.numOfSample,1))
        freq = -2 * math.pi * (shift) * self.t.T
        alpha = np.random.normal(self.parameters["met_damp"][0],sd*self.parameters["met_damp"][1],(self.numOfSample,1))
        ph = np.random.normal(self.parameters["met_ph"][0],sd*self.parameters["met_ph"][1],(self.numOfSample))
        signal = np.matmul(ampl[:, 0:self.numOfSig], self.basisset[0:self.sigLen, :].T)
        # noise = np.random.normal(0, noiseLevel, (ns, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (ns, self.sigLen))
        # signal = signal + noise

        y = np.multiply(signal, np.exp(-alpha * self.t.T))
        y = np.multiply(y, np.exp(freq * 1j))
        y += mm_signals

        y = y.T * np.exp(ph * 1j)


        noiseLevel = np.max(y) * self.parameters["noise_level"]
        noise = np.random.normal(0, noiseLevel, (self.numOfSample, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (self.numOfSample, self.sigLen))
        y = y + noise.T
        y_f = np.fft.fftshift(np.fft.fft(y[:,0:1000], axis=0), axes=0)
        self.plotsppm(y_f, 0, 5, True)
        plt.title("mean = {} and sd = {}".format(np.mean(self.cal_snrf(y_f)),np.std(self.cal_snrf(y_f))))
        self.savefig("simulated_signals")
        Path(self.sim_dir +self.parameters["child_root"]).mkdir(parents=True, exist_ok=True)
        np.savez(self.sim_dir +self.parameters["child_root"]+ "big_gaba_size_{}_sd_{}".format(self.numOfSample,sd), y , mm_signals.T, ampl, shift, alpha, ph)
        # np.save(self.sim_dir + "big_gaba_size_{}_sd_{}".format(self.numOfSample, sd), y)
    def simulation_mc(self):
        mm_signals=0
        sd = 3
        d_  = np.random.normal(self.parameters["mmd_mean"][0], sd * self.parameters["mmd_std"][0])
        d = np.random.normal(d_, 0,
                             (self.numOfSample, 1))

        for idx, (F, D, A) in enumerate(zip(self.MM_f, self.MM_d, self.MM_a)):
            f_ = np.random.normal(self.parameters["mmf_mean"][idx],sd*self.parameters["mmf_std"][idx])
            f = np.random.normal(f_,0,(self.numOfSample,1))
            a_ = np.random.normal(self.parameters["mma_mean"][idx], sd * self.parameters["mma_std"][idx])
            a = self.relu(np.random.normal(a_, 0, (self.numOfSample,1)))
            mm_signals += self.MM_model(a,f,d,0,A,self.trnfreq * F,D)


        # for idx, (F, D, A) in enumerate(zip(self.parameters["MM_f"], self.parameters["MM_d"], self.parameters["MM_d"])):
        #     ampl = np.random.normal(self.parameters["met_mean"],sd*self.parameters["met_std"],(self.numOfSample,1))

        ampl_ = np.random.normal(np.asarray(self.parameters["met_mean"]), np.asarray(self.parameters["met_std"]), size=(1, self.numOfSig))
        ampl = np.random.normal(ampl_, 0,
                                 size=(self.numOfSample, self.numOfSig))
        ampl = self.relu(ampl)
        shift_ = np.random.normal(self.parameters["met_shift"][0], sd * self.parameters["met_shift"][1])
        shift = np.random.normal(shift_,0,(self.numOfSample,1))
        freq = -2 * math.pi * (shift) * self.t.T
        alpha_ = np.random.normal(self.parameters["met_damp"][0], sd * self.parameters["met_damp"][1])
        alpha = np.random.normal(alpha_,0,(self.numOfSample,1))
        ph_ = np.random.normal(self.parameters["met_ph"][0], sd * self.parameters["met_ph"][1])
        ph = np.random.normal(ph_,0,(self.numOfSample))
        signal = np.matmul(ampl[:, 0:self.numOfSig], self.basisset[0:self.sigLen, :].T)
        # noise = np.random.normal(0, noiseLevel, (ns, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (ns, self.sigLen))
        # signal = signal + noise

        y = np.multiply(signal, np.exp(-alpha * self.t.T))
        y = np.multiply(y, np.exp(freq * 1j))
        y += mm_signals

        y = y.T * np.exp(ph * 1j)

        # noiseLevel = np.max(y) * self.parameters["noise_level"]
        # np.repeat(np.random.normal(1.6, 0.4, (self.numOfSample, 1)), 2048, 1) *
        noiseLevel = np.max(y) * self.parameters["noise_level"]
        noise = np.random.normal(0, noiseLevel, (self.numOfSample, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (self.numOfSample, self.sigLen))
        y = y + noise.T
        y_f = np.fft.fftshift(np.fft.fft(y[:, 0:1000], axis=0), axes=0)
        self.plotsppm(y_f, 0, 5, True)
        plt.title("mean = {} and sd = {}".format(np.mean(self.cal_snrf(y_f)), np.std(self.cal_snrf(y_f))))
        self.savefig("simulated_signals")
        Path(self.sim_dir + self.parameters["child_root"]).mkdir(parents=True, exist_ok=True)
        np.savez(self.sim_dir + self.parameters["child_root"] + "big_gaba_size_{}_sd_{}".format(self.numOfSample,sd), y , mm_signals.T, ampl, shift, alpha, ph)
        # np.save(self.sim_dir + "big_gaba_size_{}_sd_{}".format(self.numOfSample, sd), y)

    def getSignals(self,min_c, max_c, f, d, ph, noiseLevel, ns, mm_cond):
        if mm_cond == True:
            basisset = np.concatenate((self.basisset, self.mm), axis=1)
            numOfSig = self.numOfSig + 1
        ampl = min_c + np.multiply(np.random.random(size=(ns, numOfSig)), (max_c - min_c))
        # ampl = np.multiply(np.random.random(size=(ns, numOfSig)), (max_c))
        shift = f * np.random.rand(ns) - f / 2
        freq = -2 * math.pi * (shift) * self.t
        alpha = d * np.random.rand(ns)
        ph = (ph * np.random.rand(ns) * math.pi) - (ph / 2 * math.pi)
        signal = np.matmul(ampl[:, 0:numOfSig], basisset[0:self.sigLen, :].T)
        noise = np.random.normal(0, noiseLevel, (ns, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (ns, self.sigLen))
        signal = signal + noise

        y = np.multiply(signal, np.exp(freq * 1j).T)
        y = y.T * np.exp(ph * 1j)
        y = np.multiply(y, np.exp(-alpha * self.t))
        return y, ampl, shift, alpha, ph

    def getSignal(self,ampl, shift, alpha, ph, noiseLevel,mm_cond):
        if mm_cond == True:
            basisset = np.concatenate((self.basisset, self.mm), axis=1)
            numOfSig = self.numOfSig + 1
        freq = -2 * math.pi * (shift) * self.t
        signal = np.matmul(ampl[:, 0:numOfSig], basisset[0:self.sigLen, :].T)
        noise = np.random.normal(0, noiseLevel, (1, self.sigLen)) + 1j * np.random.normal(0, noiseLevel, (1, self.sigLen))
        y = np.multiply(signal, np.exp(freq * 1j).T)
        y = y.T * np.exp(ph * 1j)
        y = np.multiply(y, np.exp(-alpha * self.t))
        return y + noise.T, y
    def get_augment(self, signals, n, f_band, ph_band, ampl_band, d_band, noiseLevel):
        l = []
        l.append(signals)
        lens = np.shape(signals)[1]
        shift_t = f_band * np.random.rand(n * lens) - (f_band / 2)
        ph_t = ph_band * np.random.rand(n * lens) * math.pi - ((ph_band / 2) * math.pi)
        ampl_t = 1 + ((ampl_band * np.random.rand(n * lens))-ampl_band/2)
        d_t = d_band * np.random.rand(n * lens)
        for i in range(0, lens):
            signal = np.expand_dims(signals[:, i], 1)
            numOfSamplei = n
            freq = -2 * math.pi * (shift_t[i * numOfSamplei:(i + 1) * numOfSamplei]) * self.t
            ph = ph_t[i * numOfSamplei:(i + 1) * numOfSamplei]
            ampl = ampl_t[i * numOfSamplei:(i + 1) * numOfSamplei]
            d = d_t[i * numOfSamplei:(i + 1) * numOfSamplei]
            y = ampl * signal
            y = np.multiply(y * np.exp(ph * 1j), np.exp(freq * 1j))
            y = np.multiply(y, np.exp(-d * self.t))
            noise = np.random.normal(0, noiseLevel, (len(signal), numOfSamplei)) + 1j * np.random.normal(0, noiseLevel,
                                                                                                         (len(signal),
                                                                                                          numOfSamplei))
            y_i = y + noise
            l.append(y_i)
        y = np.hstack(l)
        return y, ampl_t, d_t, shift_t, ph_t
    def  savefig(self, path, plt_tight=True):
        # plt.ioff()
        if plt_tight:
            plt.tight_layout()
        if self.save:
            plt.savefig(self.saving_dir + path + ".svg", format="svg")
            # plt.savefig(self.saving_dir + path + " .png", format="png", dpi=1200)
        plt.clf()
        # plt.show()
    # %%
    # %%
    def loadModel(autoencoder, path):
        # m = LitAutoEncoder(t,signal_norm)
        return autoencoder.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    # %%
    def tic(self):
        global start_time
        start_time = time.time()
    def toc(self,name):
        elapsed_time = (time.time() - start_time)
        print("--- %s seconds ---" % elapsed_time)
        timingtxt = open(self.saving_dir + name + ".txt", 'w')
        timingtxt.write(name)
        timingtxt.write("--- %s ----" % elapsed_time)
        timingtxt.close()

    # %%
    def cal_snr(self,data, endpoints=128,offset=0):
        return np.abs(data[0, :]) / np.std(data.real[-(offset + endpoints):-(offset+1), :], axis=0)

    def cal_snrf(self,data_f,range=[2,4],endpoints=128,offset=0):
        p1 = int(self.ppm2p(range[0], data_f.shape[1]))
        p2 = int(self.ppm2p(range[1], data_f.shape[1]))
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
    def fillppm(self, y1, y2, ppm1, ppm2, rev, alpha=.1, color='red'):
        p1 = int(self.ppm2p(ppm1, len(y1)))
        p2 = int(self.ppm2p(ppm2, len(y1)))
        n = p2 - p1
        x = np.linspace(int(ppm1), int(ppm2), abs(n))
        plt.fill_between(np.flip(x), y1[p2:p1, 0].real,
                         y2[p2:p1, 0].real, alpha=alpha, color=color)
        if rev:
            plt.gca().invert_xaxis()

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

    def normalize(self,inp):
        return (np.abs(inp) / np.abs(inp).max(axis=0)) * np.exp(np.angle(inp) * 1j)

    def plotppm(self, sig, ppm1, ppm2, rev, linewidth=0.3, linestyle='-',label=None, mode='real'):
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
        g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df, linewidth=linewidth, linestyle=linestyle,label=label)
        plt.tick_params(axis='both', labelsize=fontsize)
        if rev:
            plt.gca().invert_xaxis()
        return g
        # gca = plt.plot(x,sig[p2:p1,0],linewidth=linewidth, linestyle=linestyle)

    def plot_basis2(self, basisset, ampl):

        for i in range(0, len(basisset.T) - 1):
            self.plotppm(+100* i + fft.fftshift(fft.fft(ampl * self.basisset[:, i])), 1, 7, False,label=self.met_name[i])
        self.plotppm(100 * (i + 1) + fft.fftshift(fft.fft(self.basisset[:, i + 1])), 1, 7, True,label=self.met_name[i])
        # plt.legend(self.met_name)
        self.savefig("Basis" + str(ampl),plt_tight=True)
        plt.tick_params(axis='both', labelsize=fontsize)


    def plot_basis(self, ampl, fr, damp, ph, rng=[1,5]):
        reve = False
        for i in range(0, len(self.basisset.T)):
            vv=fft.fftshift(fft.fft(ampl[0, i] * self.basisset[:len(self.t), i]*np.exp(-2 * np.pi *1j* fr * self.t.T)*np.exp(-1*damp*self.t.T)))
            if i ==len(self.basisset.T)-1:
                reve= True
            ax = self.plotppm(-4 * (i+2) + vv.T, rng[0], rng[1], reve)
            sns.despine(left=True,right=True,top=True)
            plt.text(.1, -4 * (i+2), self.met_name[i],fontsize=8)
            ax.tick_params(left=False)
            ax.set(yticklabels=[])
        plt.tick_params(axis='both', labelsize=fontsize)


    def Lornz(self, ampl, f, d, ph ,Cra, Crfr, Crd):
        return (Cra*ampl) * np.multiply(np.multiply(np.exp(ph * 1j),
                                                    np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                     np.exp(-1*(d + Crd) * self.t.T))
    def Gauss(self, ampl, f, d, ph, Cra, Crfr, Crd):
        return (Cra*ampl) * np.multiply(np.multiply(np.exp(ph * 1j),
                                                    np.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                     np.exp(-1*((d + Crd)**2) * self.t.T * self.t.T))


    def data_proc(self):
        if self.wr[0] == True:
            self.dataset = watrem.init(self.dataset[:, :], self.t_step, self.wr[1])
            with open(self.data_dir_ny, 'wb') as f:
                np.save(f, self.dataset, allow_pickle=True)
        else:
            if self.data_dir_ny is not None:
                with open(self.data_dir_ny, 'rb') as f:
                    self.dataset = np.load(f)
        if self.data_conj == True:
            y = np.conj(self.dataset)
        else:
            y = self.dataset
        # y = y[4:,:]

        y = self.normalize(y)
        ang = np.angle(y[1, :])
        y = y * np.exp(-1 * ang * 1j)
        try:
            reload = self.parameters['reload_test']
        except:
            reload = True
        if reload == True:
            with open(self.saving_dir + "test_" + str(self.test_nos), 'rb') as f:
                load = np.load(f)
                _, _, _ , indx= [load[x] for x in load]
                y_test = y[:, indx]
                not_indx = np.ones(y.shape[1], dtype=bool)
                not_indx[indx] = False
                y = y[:, not_indx]
        else:
            indx = torch.randint(self.numOfSample, (self.test_nos,))
            # subj_id = torch.arange(36,48)
            # indx = []
            # for i in subj_id:
            #     indx+=[*range(i*64,(i+1)*64)]
            y_test =  y[0:self.truncSigLen,indx]
            not_indx = np.ones(y.shape[1], dtype=bool)
            not_indx[indx] = False
            y = y[:, not_indx]
            snrs = self.cal_snr(fft.fftshift(fft.fft(y_test, axis=0), axes=0))
            data = []
            data.append(y_test)
            data.append(snrs)
            data.append(indx)
            with open(self.saving_dir + "test_" + str(self.test_nos), 'wb') as f:
                np.savez(f, *data, allow_pickle=True)
            sio.savemat(self.saving_dir + "test_" + str(self.test_nos) + "_testDB.mat",
                        {'y_test': y_test, 'snrs_t': snrs})

            Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y_test, 0), np.size(y_test, 1),
                                         self.t_step * 1000, 0, 0, self.trnfreq * 1e6), y_test,
                        self.saving_dir + self.test_name)

            Jmrui.write(Jmrui.makeHeader("basis_set", np.size(self.basisset, 0), np.size(self.basisset, 1),
                                         self.t_step * 1000, 0, 0, self.trnfreq * 1e6), self.basisset,
                        self.saving_dir + "basis_set")
            np.save(self.saving_dir + "basis_set", self.basisset)
        self.y_test_idx = indx
        # y_f = fft.fftshift(fft.fft(y, axis=0), axes=0)

        # if self.quality_filt[0] == True:
        #     cond = np.mean(np.abs(y_f[self.quality_filt[1]:self.quality_filt[2], :]), axis=0)
        #     # con2 = np.mean(np.abs(y_f[self.quality_filt[3]:self.quality_filt[4], :]), axis=0)
        #     idx = np.where((cond < 6))[0]
        #     y = y[0:2 * self.truncSigLen, idx]
        #     cond = np.mean(np.abs(y_f[self.quality_filt[1]:self.quality_filt[2], :]), axis=0)
        #     self.y_test_idx = np.where((cond < 6))[0]
        return y,y_test

    def data_prep(self):
        if self.parameters["simulated"]:
            data = np.load(self.data_dir_ny)
            y , _, self.ampl_t, _, self.alpha, _ = [data[x] for x in data]
        else:
            y, self.y_test = self.data_proc()
        if self.aug_params is not None:
            y, _, _, _, _ = self.data_aug(y[0:self.sigLen,:])
        y_f = fft.fftshift(fft.fft(y, axis=0),axes=0)
        if self.pre_plot ==True:
            plt.hist(self.cal_snrf(y_f))
            plt.show()
            self.plotppm(fft.fftshift(fft.fft((y[:, 2000]), axis=0)), 1, 8, True, linewidth=1, linestyle='-')
            plt.show()

        self.numOfSample = np.shape(y)[1];
        y_norm = y
        del y
        self.to_tensor(y_norm)

    def data_aug(self,y):
        return self.get_augment(y, self.aug_params[0], self.aug_params[1], self.aug_params[2], self.aug_params[3], self.aug_params[4], self.aug_params[5])
    def to_tensor(self,y_norm):
        y_trun = y_norm[0:self.truncSigLen, :].astype('complex64')
        self.y_trun = torch.from_numpy(y_trun[:, 0:self.numOfSample].T)
        if self.parameters["simulated"] is False:
            y_test = self.y_test[0:self.truncSigLen, :].astype('complex64')
            self.y_test_trun = torch.from_numpy(y_test[:, 0:self.numOfSample].T)
            self.train = TensorDataset(self.y_trun)
            self.val = TensorDataset(self.y_test_trun)
        else:
            self.y_test_trun = self.y_trun
            labels = torch.from_numpy(np.hstack((self.ampl_t,self.alpha)))
            labels = labels.type(torch.float32)
            my_dataset = TensorDataset(self.y_trun,labels)
            self.train, self.val = random_split(my_dataset, [int((self.numOfSample) * self.tr_prc), self.numOfSample - int((self.numOfSample) * self.tr_prc)])


    def inputSig(self,x,p1=None,p2=None):
        if self.parameters['domain'] == 'freq':
            if self.parameters['zero_fill'][0] == True:
                x = self.zero_fill_torch(x,1,self.parameters['zero_fill'][1])
            x = torch.fft.fftshift(torch.fft.fft(x, dim=1), dim=1)
            if p1==None and p2==None:
                p1 = int(self.ppm2p(self.parameters['fbound'][2], (self.truncSigLen)))
                p2 = int(self.ppm2p(self.parameters['fbound'][1], (self.truncSigLen)))

            x = x[:, p1:p2]
            if self.in_shape == 'complex':
                return torch.cat((torch.unsqueeze(x.real, 1), torch.unsqueeze(x.imag, 1)),1)
            if self.in_shape == 'real':
                return torch.unsqueeze(x.real, 1)
        else:
            if self.in_shape == 'complex':
                return torch.cat((torch.unsqueeze(x[:, 0:self.truncSigLen].real, 1),
                                  torch.unsqueeze(x[:, 0:self.truncSigLen].imag, 1)), 1)
            if self.in_shape == 'real':
                return torch.unsqueeze(x[:, 0:self.truncSigLen].real, 1)

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


    def test_compact(self,ensembles = True,crlb=True,estimate=False):
        id = self.test_data_root
        if self.parameters['test_load']:
            data = np.load(self.sim_dir + id + '.npz')
            y_test, mm_signals, ampl_t, shift_t, alpha_t, ph_t = [data[x] for x in data]
            snrs = self.cal_snrf(fft.fftshift(fft.fft(y_test, axis=0), axes=0))
            mm_signals = mm_signals * np.exp(1*ph_t)
            # Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y_test, 0), np.size(y_test, 1), self.t_step * 1000, 0, 0,
            #                              self.trnfreq), y_test,
            #             self.sim_dir+ id + '_testDBjmrui.txt')
            # sio.savemat(self.sim_dir+ id + "_testDB.mat",
            #             {'y_test': y_test, 'ampl_t': ampl_t, 'shift_t': shift_t, 'alpha_t': alpha_t, 'ph_t': ph_t,'snrs_t': snrs, })
            #
            # Jmrui.write(Jmrui.makeHeader("basis", np.size(self.basisset[0:self.sigLen, :], 0), np.size(self.basisset[0:self.sigLen, :], 1), self.t_step * 1000, 0, 0,
            #                              self.trnfreq), self.basisset[0:self.sigLen, :],
            #             self.sim_dir+"basisset.txt")
        else:
            # min_c, max_c, f, d, ph, noiseLevel, ns, mm_cond
            data = self.getSignals(self.test_params[0], self.test_params[1], self.test_params[2]
                       ,self.test_params[3], self.test_params[4], self.test_params[5],
                       self.test_params[6], True)
            y_test, ampl_t, shift_t, alpha_t, ph_t = data
            snrs = self.cal_snrf(fft.fftshift(fft.fft(y_test, axis=0),axes=0))
            data=list(data)
            data.append(snrs)
            np.savez(self.sim_dir+"test_"+str(self.test_params[2:]),*data)
            Jmrui.write(Jmrui.makeHeader("tesDB", np.size(y_test, 0), np.size(y_test, 1), self.t_step * 1000, 0, 0,
                                         self.trnfreq), y_test,
                        self.sim_dir +"test_" + str(self.test_params[2:]) + '_testDBjmrui.txt')
            sio.savemat(self.sim_dir+"test_"+str(self.test_params[2:]) + "_testDB.mat",
                        {'y_test': y_test, 'ampl_t': ampl_t, 'shift_t': shift_t, 'alpha_t': alpha_t, 'ph_t': ph_t,'snrs_t': snrs, })

        id = "test/" + id + "/"
        selected_met = ["Cr", "GPC", "Glu", "Ins", "NAA", "NAAG", "PCho", "PCr", "Tau"]
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)
        with open(self.saving_dir + id  + "snrs.txt",'w') as f:
            f.write("snr mean i/s {} and snr sd is {}".format(np.mean(snrs),np.std(snrs)))
        y_test = y_test.astype('complex64')
        y_test = torch.from_numpy(y_test)
        # print(y_test.size())
        self.tic()

        # autoencoder = self.autoencoders[0]
        # if crlb == True:
        #     crlb = []
        #     for i in y_test[:,0:5].T:
        #         crlb.append(self.getCRLB(autoencoder, i.T.to(self.parameters["gpu"])).cpu().detach().numpy())
        #     crlb = pd.DataFrame(np.asarray(crlb)[:,0,0:self.numOfSig],columns=self.met_name)
        #     crlb.to_csv(self.saving_dir + id + "_" + "crlb.csv")
        if estimate:
            if ensembles == True:
                rslt = self.predict_ensembles(y_test)
            else:
                rslt = self.predict_MCDO(y_test)

            self.toc(id + "time")
            y_out, mean_, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc, decoded_net, mm_ = rslt
            autoencoder=self.autoencoders[0]
            if crlb == True:
                crlb_ = []
                for i , j in zip(y_test[:,:].T,mean_):
                    crlb_.append(self.getCRLB(autoencoder, i.T.to(self.parameters["gpu"]), ampl=torch.from_numpy(j).to(self.parameters["gpu"]).unsqueeze(0)).cpu().detach().numpy())
                aleatoric_unc=(np.asarray(crlb_)[:, 0:self.numOfSig])
            else:
                aleatoric_unc = epistemic_unc
                np.savez(self.saving_dir + id + "rslt_wiithout_ph_1.npz", y_out, mean_, fr, damp, ph, decs, encs, epistemic_unc,
                         aleatoric_unc, decoded_net, mm_)
        else:
            rslt = np.load(self.saving_dir + id + 'rslt_wiithout_ph_1.npz', allow_pickle=True)
            y_out, mean_, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc, decoded_net, mm_ = [rslt[x] for x in rslt]
        # y_out, mean_, fr, damp, ph, decs, encs,epistemic_unc, aleatoric_unc,decoded_net,mm_ = rslt_wiithout_ph_1
        # test_info[self.met_name] = np.abs((ampl_t - y_out)/(ampl_t + y_out))*100
        if self.MM == True:
            self.numOfComp = self.numOfSig +1


        epistemic_unc_name = [i + "_epistemic_unc" for i in self.met_name]
        aleatoric_unc_name= [i + "_aleatoric_unc" for i in self.met_name]
        test_info = pd.DataFrame()
        test_info['SNR'] = snrs
        # if shift_t.shape == (1,1):
        #     test_info['Frequency'] = shift_t[0,0]
        #     test_info['Damping'] = alpha_t[0,0]
        #     test_info['Phase'] = ph_t[0]
        #     test_info[self.met_name] = ampl_t[0, 0:len(self.met_name)]
        # else:
        test_info['Frequency'] = shift_t
        test_info['Damping'] = alpha_t
        test_info['Phase'] = ph_t
        test_info[self.met_name] = ampl_t[:, 0:len(self.met_name)]
        test_info[epistemic_unc_name] = 0
        test_info[aleatoric_unc_name] = 0
        test_info['type'] = 'True'
        test_temp = pd.DataFrame()
        test_temp['SNR'] = snrs
        test_temp['Frequency'] = fr
        test_temp['Damping'] = damp
        test_temp['Phase'] = ph
        test_temp[self.met_name] = y_out
        test_temp['type'] = 'Predicted'
        test_info[epistemic_unc_name] = epistemic_unc
        test_info[aleatoric_unc_name] = aleatoric_unc
        test_info = test_info.append(test_temp)
        test_info.to_csv(self.saving_dir + id + "rslt_wiithout_ph_1.csv")

        test_temp = pd.DataFrame()
        test_temp[epistemic_unc_name] = np.expand_dims(epistemic_unc.mean(0),0)
        test_temp[aleatoric_unc_name] = np.expand_dims(aleatoric_unc.mean(0),0)
        test_temp.to_csv(self.saving_dir + id + "uncertain_avg.csv")

        errors_DL = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2','intercept','coef'], index=self.met_name)
        for i in range(0, self.ens-1):
            for j in range(0, self.numOfSig):
                # ax = sns.regplot(x=ampl_t[:, j], y=encs[i, :, j],label=str(i))
                model = LinearRegression().fit(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1)))
                errors_DL.iloc[j] = [r2_score(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_squared_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_percentage_error(ampl_t[:, j], encs[i, :, j]) * 100,
                                     model.score(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1))),
                                     model.intercept_[0],
                                     model.coef_[0][0]]
            errors_DL.to_csv(self.saving_dir + id + "_" + str(i) + "Ens_errorsDL.csv")
        # y_out, y_out_var, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc, decoded_net, mm_
        rang = [0,5]
        mm_f = fft.fftshift(fft.fft(mm_[:, :, :], axis=2), axes=2)
        y_out_f = fft.fftshift(fft.fft(decs[:, :, :], axis=2), axes=2)
        decoded_net = self.zero_fill(decoded_net[:, :, 0, :] + 1j * decoded_net[:, :, 1, :],2,2048)
        decoded_net_f = fft.fftshift(fft.fft(decoded_net[:,:,:], axis=2), axes=2)
        ampl = y_out
        shift = fr
        damp = damp
        ph = ph
        sd_f = 5
        idx = 11
        self.plotppm(40 + np.fft.fftshift(np.fft.fft((y_test[0:2048, idx])).T), rang[0], rang[1], False, linewidth=0.3, linestyle='-')
        # rec_signal, _, enc, fr, damp, ph, mm_v, _ = self.testmodel(self.autoencoders[0], y_test.T.cuda())
        self.plotppm(15+mm_f[0,idx, :], rang[0],rang[1], False, linewidth=1, linestyle='-')
        self.fillppm(15+np.expand_dims(mm_f[0,idx, :] - sd_f * np.std(mm_f[1:,idx, :], 0),axis=1),
                     15+np.expand_dims(mm_f[0,idx, :] + sd_f * np.std(mm_f[1:,idx, :], 0),axis=1), 0, 5, False, alpha=.1,color='orange')



        self.plotppm(30+
        y_out_f[0,idx, :], rang[0],
            rang[1], False, linewidth=1, linestyle='-')
        self.fillppm(30+np.expand_dims(y_out_f[0,idx, :] - sd_f * np.std(y_out_f[1:,idx, :], 0),axis=1),
                     30+np.expand_dims(y_out_f[0,idx, :] + sd_f * np.std(y_out_f[1:,idx, :], 0),axis=1), 0, 5, False, alpha=.1,color='green')


        self.plotppm(20+
        decoded_net_f[0,idx, :], rang[0],
            rang[1], False, linewidth=1, linestyle='-')
        self.fillppm(20+np.expand_dims(decoded_net_f[0,idx, :]- sd_f * np.std(decoded_net_f[1:,idx, :], 0),axis=1),
                     20+np.expand_dims(decoded_net_f[0,idx, :] + sd_f * np.std(decoded_net_f[1:,idx, :], 0),axis=1), 0, 5, False, alpha=.1,color='red')

        self.plotppm(5 +
            np.fft.fftshift(np.fft.fft((y_test[0:2048, idx]))) - y_out_f[0,idx, :],
                     rang[0], rang[1],
                     False, linewidth=1, linestyle='-')
        self.fillppm(5+np.expand_dims(- sd_f * np.std(y_out_f[1:,idx, :], 0),axis=1),
                     5+np.expand_dims(+ sd_f * np.std(y_out_f[1:,idx, :], 0),axis=1), 0, 5, False, alpha=.1,color='violet')

        self.plotppm(
            np.fft.fftshift(np.fft.fft((y_test[0:2048, idx]))) - decoded_net_f[0,idx, :],
                     rang[0], rang[1],
                     False, linewidth=1, linestyle='-')
        self.fillppm(np.expand_dims(- sd_f * np.std(decoded_net_f[1:,idx, :], 0),axis=1),
                     np.expand_dims(+ sd_f * np.std(decoded_net_f[1:,idx, :], 0),axis=1), 0, 5, False, alpha=.1,color='violet')

        self.plotppm(
            10+ np.fft.fftshift(np.fft.fft((mm_signals[0:2048, idx]))) - mm_f.mean(0)[idx, :],
            rang[0], rang[1],
            True, linewidth=1, linestyle='-')

        sns.despine()
        self.plot_basis(np.expand_dims(ampl[idx, :],axis=0), shift[idx, :],
                        damp[idx, :],
                        ph[idx, :])
        self.savefig(id  +"_tstasig")


    def test(self,ensembles = True, crlb=False):
        cmap = 'Blues'
        id = self.test_data_root
        data = np.load(self.sim_dir + id + '.npz')
        y_test, mm_signals, ampl_t, shift_t, alpha_t, ph_t = [data[x] for x in data]
        snrs = self.cal_snrf(fft.fftshift(fft.fft(y_test, axis=0), axes=0))
        id_test = "test/" + id + "/"
        id = "test_all/" + id + "/"
        selected_met = ["Cr", "GPC", "Glu", "mIns", "NAA", "NAAG", "PCho", "PCr", "Tau"]
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)
        test_info = pd.DataFrame()
        test_info['SNR'] = snrs
        test_info['Frequency'] = shift_t
        test_info['Damping'] = alpha_t
        test_info['Phase'] = ph_t

        sns.set(style="white", palette="muted", color_codes=True)
        sns.distplot(test_info['SNR'], color="m")
        sns.despine()
        self.savefig("test_snr_hist")
        
        y_test = y_test.astype('complex64')
        y_test = torch.from_numpy(y_test)
        # self.tic()
        # if ensembles == True:
        #     y_out, mean_, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc,decoded_net,mm_ = self.predict_ensembles(y_test)
        # else:
        #     y_out, mena_, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc, decoded_net, mm_ = self.predict_MCDO(
        #         y_test)
        # self.toc(id + "time")
        rslt = np.load(self.saving_dir + id_test + 'rslt_wiithout_ph_1.npz', allow_pickle=True)
        y_out, mean_, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc,decoded_net,mm_ = [rslt[x] for x in rslt]
        # autoencoder=self.autoencoders[0]
        # if crlb == True:
        #     crlb_ = []
        #     for i , j in zip(y_test[:,:].T,mean_):
        #         crlb_.append(self.getCRLB(autoencoder, i.T.to(self.parameters["gpu"]), ampl=torch.from_numpy(j).to(self.parameters["gpu"]).unsqueeze(0)).cpu().detach().numpy())
        # aleatoric_unc=(np.asarray(crlb_)[:, 0:self.numOfSig])


        # test_info[self.met_name] = np.abs((ampl_t - y_out)/(ampl_t + y_out))*100
        if self.MM == True:
            self.numOfComp = self.numOfSig +1
        test_info[self.met_name] = np.abs(ampl_t[:,0:len(self.met_name)] - y_out)
        net_ale = pd.DataFrame(aleatoric_unc, columns=self.met_name)
        net_epis = pd.DataFrame(epistemic_unc, columns=self.met_name)

        sns.stripplot(data=net_epis[['Cr', 'NAA', 'PCr']])
        self.savefig(id + "stripplot_epis")

        sns.stripplot(data=net_ale[['Cr', 'NAA', 'PCr']])
        self.savefig(id + "stripplot_ale")

        def summarize(df, name):
            mean = df.mean()
            std = df.std()
            smry = pd.DataFrame({'Mean':mean, 'std':std})
            smry.to_csv(self.saving_dir + id + name + '.csv')

        summarize(net_ale, 'summary_aleatoric')
        summarize(net_epis, 'summary_epistemic')

        type = ['Predicted' for i in y_out]
        net_pred = pd.DataFrame(y_out,columns=self.met_name)
        net_pred['type'] = type
        type = ['True' for i in y_out]
        net_true = pd.DataFrame(ampl_t[:,0:len(self.met_name)],columns=self.met_name)
        net_true['type'] = type
        net_pred = net_pred.append(net_true)
        dfm = pd.melt(net_pred, id_vars=['type'])

        lc = [self.met_name[i] for i in (np.where(self.max_c[0:len(self.met_name)] < 0.3)[0])]
        sns.set_style('whitegrid')
        sns.violinplot(x='variable', y='value', data=dfm[dfm['variable'].isin(lc)], hue='type', palette="Set3",
                       linewidth=1,
                       split=True,
                       inner="quartile")
        sns.despine()
        self.savefig(id + "violion_low")
        
        lc = [self.met_name[i] for i in (np.where((self.max_c[0:len(self.met_name)]  > 0.3) & (self.max_c[0:len(self.met_name)]  < 1.01))[0])]
        sns.violinplot(x='variable', y='value', data=dfm[dfm['variable'].isin(lc)], hue='type', palette="Set3",
                       linewidth=1,
                       split=True,
                       inner="quartile")
        sns.despine()
        self.savefig(id + "violion_high")
        
        corr = test_info.corr()
        corr.iloc[4:, 0:4].transpose().to_csv(self.saving_dir + id + "_errors_corr.csv")
        sns.heatmap(data=corr.iloc[4:, 0:4].transpose(), cmap=self.heatmap_cmap)
        self.savefig(id + "corrollation_heatmap")

        errors_DL = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2','intercept','coef'], index=self.met_name)
        for i in range(0, self.ens-1):
            for j in range(0, self.numOfSig):
                model = LinearRegression().fit(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1)))
                errors_DL.iloc[j] = [r2_score(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_squared_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_percentage_error(ampl_t[:, j], encs[i, :, j]) * 100,
                                     model.score(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1))),
                                     model.intercept_[0],
                                     model.coef_[0][0]]

            errors_DL.to_csv(self.saving_dir + id + "_" + str(i) + "Ens_errorsDL.csv")

        file = open(self.saving_dir + id + '_predicts.csv', 'w')
        writer = csv.writer(file)
        writer.writerows(np.concatenate((y_out, epistemic_unc,aleatoric_unc, fr, damp, ph), axis=1))
        file.close()
        mean_f = np.mean((fr) - np.expand_dims(shift_t, axis=[1]))
        mean_alph = np.mean((damp) - np.expand_dims(alpha_t, axis=[1]))
        mean_ph = np.mean((ph) - np.expand_dims(ph_t, axis=[1]))
        std_f = np.std((fr) - np.expand_dims(shift_t, axis=[1]))
        std_alph = np.std((damp) - np.expand_dims(alpha_t, axis=[1]))
        std_ph = np.std((ph) - np.expand_dims(ph_t, axis=[1]))

        file = open(self.saving_dir + id + '_rslt.csv', 'w')
        writer = csv.writer(file)
        writer.writerow(["freq", mean_f, std_f])
        writer.writerow(["damp", mean_alph, std_alph])
        writer.writerow(["ph", mean_ph, std_ph])
        
        ax = self.modified_bland_altman_plot(shift_t, fr, gt=shift_t)
        self.savefig(id + "freq")



        ax = self.modified_bland_altman_plot(alpha_t, damp,gt=alpha_t)
        self.savefig(id + "damp")

        ax = self.modified_bland_altman_plot(ph_t * 180 / np.pi, ph[:, 0] * 180 / np.pi,gt = ph_t * 180 / np.pi)
        self.savefig(id + "ph")

        ids1 = [2, 12, 8, 14, 17, 9]
        ids2 = [15, 13, 7, 5, 6, 10]
        names = ["Cr+PCr", "NAA+NAAG", "Glu+Gln", "PCho+GPC", "Glc+Tau", "Ins+Gly"]
        errors_combined = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2', 'intercept', 'coef'], index=names)
        idx = 0
        for id1, id2, name in zip(ids1, ids2, names):
            self.modified_bland_altman_plot(ampl_t[:, id1] + ampl_t[:, id2], (y_out[:, id1] + y_out[:, id2]),gt=ampl_t[:, id1] + ampl_t[:, id2])
            plt.title(name)
            self.savefig(id + "combined_" + name)
            model = LinearRegression().fit((ampl_t[:, id1] + ampl_t[:, id2]).reshape((-1, 1)),
                                           (y_out[:, id1] + y_out[:, id2]).reshape((-1, 1)))
            errors_combined.iloc[idx] = [r2_score(ampl_t[:, id1] + ampl_t[:, id2], (y_out[:, id1] + y_out[:, id2])),
                                         mean_absolute_error(ampl_t[:, id1] + ampl_t[:, id2],
                                                             (y_out[:, id1] + y_out[:, id2])),
                                         mean_squared_error(ampl_t[:, id1] + ampl_t[:, id2],
                                                            (y_out[:, id1] + y_out[:, id2])),
                                         mean_absolute_percentage_error(ampl_t[:, id1] + ampl_t[:, id2],
                                                                        (y_out[:, id1] + y_out[:, id2])) * 100,
                                         model.score((ampl_t[:, id1] + ampl_t[:, id2]).reshape((-1, 1)),
                                                     (y_out[:, id1] + y_out[:, id2]).reshape((-1, 1))),
                                         model.intercept_,
                                         model.coef_
                                         ]
            idx += 1
        errors_combined.to_csv(self.saving_dir + id + "_errors_combined.csv")
        errors_averaged = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2', 'intercept', 'coef'],
                                       index=self.met_name)

        if self.parameters["detailed_test"]:
            j = 0
            errors_corr = pd.DataFrame(columns=self.met_name, index=["damping", 'frequency', 'Phase', 'SNR'])
            for idx, name in enumerate(self.met_name):
                # idx = self.met_name.index(name)
                model = LinearRegression().fit(ampl_t[:, idx].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1)))
                errors_averaged.iloc[idx] = [r2_score(ampl_t[:, idx], y_out[:, idx]),
                                           mean_absolute_error(ampl_t[:, idx], y_out[:, idx]),
                                           mean_squared_error(ampl_t[:, idx], y_out[:, idx]),
                                           mean_absolute_percentage_error(ampl_t[:, idx], y_out[:, idx]) * 100,
                                           model.score(ampl_t[:, idx].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1))),
                                           model.intercept_,
                                           model.coef_]

            for idx, name in enumerate(selected_met):
                idx = self.met_name.index(name)
                # model = LinearRegression().fit(ampl_t[:, idx].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1)))
                # errors_averaged.iloc[j] = [r2_score(ampl_t[:, idx], y_out[:, idx]),
                #                            mean_absolute_error(ampl_t[:, idx], y_out[:, idx]),
                #                            mean_squared_error(ampl_t[:, idx], y_out[:, idx]),
                #                            mean_absolute_percentage_error(ampl_t[:, idx], y_out[:, idx]) * 100,
                #                            model.score(ampl_t[:, idx].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1))),
                #                            model.intercept_,
                #                            model.coef_
                #                            ]

                # yerr = 100 * np.abs(np.sqrt(y_out_var[:, idx]) / y_out[:, idx])
                yerr_ale = 100 * np.abs((aleatoric_unc[:, idx]) / y_out[:, idx])
                yerr_epis = 100 * np.abs((epistemic_unc[:, idx]) / y_out[:, idx])
                # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt = ampl_t[:, idx], c_map=cmap,c=yerr)
                # plt.title(name)
                # self.savefig(id + "seperated_percent" + name)

                self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt = ampl_t[:, idx], c_map='Reds', c=yerr_ale)
                plt.title(name)
                self.savefig(id + "seperated_percent_ale" + name)

                self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt = ampl_t[:, idx], c_map='Greens', c=yerr_epis)
                plt.title(name)
                self.savefig(id + "seperated_percent_epis" + name)

                # yerr = np.abs(np.sqrt(y_out_var[:, idx]))
                yerr_ale = np.abs((aleatoric_unc[:, idx]))
                yerr_epis = np.abs((epistemic_unc[:, idx]))
                # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt = ampl_t[:, idx], c_map=cmap, c=yerr)
                # plt.title(name)
                # self.savefig(id + "seperated" + name)

                self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt = ampl_t[:, idx], c_map='Reds', c=yerr_ale)
                plt.title(name)
                self.savefig(id + "seperated_ale" + name)

                self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt = ampl_t[:, idx], c_map='Greens', c=yerr_epis)
                plt.title(name)
                self.savefig(id + "seperated_epis" + name)

                j += 1
            list_ = [snrs,shift_t, alpha_t, ph_t]
            list_i = ["snrs_","fr_","dampings_","phase_"]
            for idx_null, name in enumerate(selected_met):
                for x,name_ in zip(list_,list_i):
                    idx = self.met_name.index(name)
                    err = np.abs(y_out[:, idx] - ampl_t[:, idx])
                    # yerr = 100 * np.abs(np.sqrt(y_out_var[:, idx]) / y_out[:, idx])

                    # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=x,  c_map=cmap, c=yerr)
                    # plt.title(name)
                    # self.savefig(id + "corrollation_precent_" + name_ + name)


                    self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=x, c_map='Reds', c=yerr_ale)
                    plt.title(name)
                    self.savefig(id + "corrollation_yerr_ale_precent_" + name_ + name)

                    self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=x, c_map='Greens', c=yerr_epis)
                    plt.title(name)
                    self.savefig(id + "corrollation_yerr_epis_precent_" + name_ + name)

                    # yerr = np.abs((y_out_var[:, idx]))
                    yerr_ale = np.abs((aleatoric_unc[:, idx]))
                    yerr_epis = np.abs((epistemic_unc[:, idx]))

                    # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=x, c_map=cmap, c=yerr)
                    # plt.title(name)
                    # self.savefig(id + "corrollation_" + name_ + name)


                    self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=x, c_map='Reds', c=yerr_ale)
                    plt.title(name)
                    self.savefig(id + "corrollation_yerr_ale_" + name_ + name)

                    self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=x, c_map='Greens', c=yerr_epis)
                    plt.title(name)
                    self.savefig(id + "corrollation_yerr_epis_" + name_ + name)

                j += 1


            errors_averaged.to_csv(self.saving_dir + id + "_errors_averaged.csv")
            name = 'Cr'
            idx = self.met_name.index(name)
            yerr_ale = 100 * np.abs((aleatoric_unc[:, idx]) / y_out[:, idx])
            yerr_epis = 100 * np.abs((epistemic_unc[:, idx]) / y_out[:, idx])

            plt.scatter(shift_t, yerr_epis)
            self.savefig(id + "yerr_pres_epis_vs_" + 'fr_' + name)

            plt.scatter(snrs, yerr_ale)
            self.savefig(id + "yerr_pres_ale_vs_" + 'snr_' + name)

            plt.scatter(alpha_t, yerr_ale)
            self.savefig(id + "yerr_pres_ale_vs_" + 'damp_' + name)

            plt.scatter(ampl_t[:, idx]-y_out[:, idx],yerr_epis)
            self.savefig(id + "yerr_pres_epis_vs_" + 'error_' + name)

            name = 'NAA'
            idx = self.met_name.index(name)
            yerr_ale = 100 * np.abs((aleatoric_unc[:, idx]) / y_out[:, idx])
            yerr_epis = 100 * np.abs((epistemic_unc[:, idx]) / y_out[:, idx])
            plt.scatter(ampl_t[:, idx]-y_out[:, idx],yerr_epis)
            self.savefig(id + "yerr_pres_epis_vs_" + 'error_' + name)
        # file.close()


    def test_MonteCarlo(self,ensembles = True):
        cmap = 'Blues'
        id = self.test_data_root
        data = np.load(self.sim_dir + id + '.npz')
        y_test, mm_signals, ampl_t, shift_t, alpha_t, ph_t = [data[x] for x in data]
        snrs = self.cal_snrf(fft.fftshift(fft.fft(y_test, axis=0), axes=0))

        # data = np.load(self.sim_dir + "test_" + str(self.test_params[2:]) + '.npz')
        # y_test, ampl_t, shift_t, alpha_t, ph_t, snrs = [data[x] for x in data]
        id_test = "test/" + id + "/"
        id = "test_all/" + id + "/"
        selected_met = ["Cr", "GPC", "Glu", "mIns", "NAA", "NAAG", "PCho", "PCr", "Tau"]
        Path(self.saving_dir + id).mkdir(parents=True, exist_ok=True)
        test_info = pd.DataFrame()
        test_info['SNR'] = snrs
        test_info['Frequency'] = shift_t
        test_info['Damping'] = alpha_t
        test_info['Phase'] = ph_t

        sns.set(style="white", palette="muted", color_codes=True)
        sns.distplot(test_info['SNR'], color="m")
        sns.despine()
        self.savefig("test_snr_hist")

        y_test = y_test.astype('complex64')
        y_test = torch.from_numpy(y_test)
        # print(y_test.size())
        rslt = np.load(self.saving_dir + id_test + 'rslt_wiithout_ph_1.npz', allow_pickle=True)
        y_out, mean_, fr, damp, ph, decs, encs, epistemic_unc, aleatoric_unc, decoded_net, mm_ = [rslt[x] for x in rslt]

        # test_info[self.met_name] = np.abs((ampl_t - y_out)/(ampl_t + y_out))*100
        if self.MM == True:
            self.numOfComp = self.numOfSig + 1
        test_info[self.met_name] = np.abs(ampl_t[:, 0:len(self.met_name)] - y_out)
        type = ['Predicted' for i in y_out]
        net_pred = pd.DataFrame(y_out, columns=self.met_name)
        net_pred['type'] = type
        type = ['True' for i in y_out]
        net_true = pd.DataFrame(ampl_t[:, 0:len(self.met_name)], columns=self.met_name)
        net_true['type'] = type
        net_pred = net_pred.append(net_true)

        corr = test_info.corr()
        corr.iloc[4:, 0].transpose().to_csv(self.saving_dir + id + "_errors_corr.csv")
        sns.heatmap(data=corr.iloc[4:, 0:1].transpose(), cmap=self.heatmap_cmap)
        self.savefig(id + "corrollation_heatmap")

        # quest, true = rslt_vis.getQuest()
        errors_DL = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2', 'intercept', 'coef'], index=self.met_name)
        errors_Q = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2', 'intercept', 'coef'], index=self.met_name)
        for i in range(0, self.ens-1):
            for j in range(0, self.numOfSig):
                # ax = sns.regplot(x=ampl_t[:, j], y=encs[i, :, j],label=str(i))
                model = LinearRegression().fit(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1)))
                errors_DL.iloc[j] = [r2_score(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_squared_error(ampl_t[:, j], encs[i, :, j]),
                                     mean_absolute_percentage_error(ampl_t[:, j], encs[i, :, j]) * 100,
                                     model.score(ampl_t[:, j].reshape((-1, 1)), encs[i, :, j].reshape((-1, 1))),
                                     model.intercept_[0],
                                     model.coef_[0][0]]
                # plt.title("DL::" + str(i) + "::"+ self.met_name[j])
                # # plt.legend()
                # x0, x1 = ax.get_xlim()
                # y0, y1 = ax.get_ylim()
                # lims = [max(x0, y0), min(x1, y1)]
                # ax.plot(lims, lims, '--k')
                # plt.show()
            errors_DL.to_csv(self.saving_dir + id + "_" + str(i) + "Ens_errorsDL.csv")

        file = open(self.saving_dir + id + '_predicts.csv', 'w')
        writer = csv.writer(file)
        writer.writerows(np.concatenate((y_out, mean_, epistemic_unc, aleatoric_unc, fr, damp, ph), axis=1))
        file.close()
        mean_f = np.mean((fr) - np.expand_dims(shift_t, axis=[1]))
        mean_alph = np.mean((damp) - np.expand_dims(alpha_t, axis=[1]))
        mean_ph = np.mean((ph) - np.expand_dims(ph_t, axis=[1]))
        std_f = np.std((fr) - np.expand_dims(shift_t, axis=[1]))
        std_alph = np.std((damp) - np.expand_dims(alpha_t, axis=[1]))
        std_ph = np.std((ph) - np.expand_dims(ph_t, axis=[1]))

        file = open(self.saving_dir + id + '_rslt.csv', 'w')
        writer = csv.writer(file)
        writer.writerow(["freq", mean_f, std_f])
        writer.writerow(["damp", mean_alph, std_alph])
        writer.writerow(["ph", mean_ph, std_ph])
        # test_info['f_error'] = fr[:, 0] - np.expand_dims(shift_t, axis=[1])[:, 0]
        # test_info['d_error'] = damp[:, 0] - np.expand_dims(alpha_t, axis=[1])[:, 0]
        # test_info['p_error'] = (ph[:, 0] * 180 / np.pi) - np.expand_dims(ph_t * 180 / np.pi, axis=[1])[:, 0]
        # sns.jointplot(x=test_info[['f_error','d_error']], y=test_info[['f','d']])

        ax = self.modified_bland_altman_plot(shift_t, fr)
        self.savefig(id + "freq")

        ax = self.modified_bland_altman_plot(alpha_t, damp)
        self.savefig(id + "damp")

        ax = self.modified_bland_altman_plot(ph_t * 180 / np.pi, ph[:, 0] * 180 / np.pi)
        self.savefig(id + "ph")

        ids1 = [2, 12, 8, 14, 17, 9]
        ids2 = [15, 13, 7, 5, 6, 10]
        names = ["Cr+PCr", "NAA+NAAG", "Glu+Gln", "PCho+GPC", "Glc+Tau", "Ins+Gly"]
        errors_combined = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2', 'intercept', 'coef'], index=names)
        idx = 0
        for id1, id2, name in zip(ids1, ids2, names):
            # var = (y_out_var[:, id1]**2 + y_out_var[:, id2]**2) + (ampl_t[:, id1]**2 + ampl_t[:, id2]**2)
            # corr, _ = pearsonr(ampl_t[:, id1], ampl_t[:, id2])
            # warning! how we can calculate sd for two corrolated normal distribution!?
            # sd = 100 * np.sqrt(y_out_var[:, id1] + y_out_var[:, id2]) / (y_out[:, id1] + y_out[:, id2])
            self.modified_bland_altman_plot(ampl_t[:, id1] + ampl_t[:, id2], (y_out[:, id1] + y_out[:, id2]))

            plt.title(name)
            self.savefig(id + "combined_" + name)

            model = LinearRegression().fit((ampl_t[:, id1] + ampl_t[:, id2]).reshape((-1, 1)),
                                           (y_out[:, id1] + y_out[:, id2]).reshape((-1, 1)))
            errors_combined.iloc[idx] = [r2_score(ampl_t[:, id1] + ampl_t[:, id2], (y_out[:, id1] + y_out[:, id2])),
                                         mean_absolute_error(ampl_t[:, id1] + ampl_t[:, id2],
                                                             (y_out[:, id1] + y_out[:, id2])),
                                         mean_squared_error(ampl_t[:, id1] + ampl_t[:, id2],
                                                            (y_out[:, id1] + y_out[:, id2])),
                                         mean_absolute_percentage_error(ampl_t[:, id1] + ampl_t[:, id2],
                                                                        (y_out[:, id1] + y_out[:, id2])) * 100,
                                         model.score((ampl_t[:, id1] + ampl_t[:, id2]).reshape((-1, 1)),
                                                     (y_out[:, id1] + y_out[:, id2]).reshape((-1, 1))),
                                         model.intercept_,
                                         model.coef_
                                         ]

            idx += 1
        errors_combined.to_csv(self.saving_dir + id + "_errors_combined.csv")
        errors_averaged = pd.DataFrame(columns=['R2', 'MAE', 'MSE', 'MAPE', 'r2', 'intercept', 'coef'],
                                       index=self.met_name)

        if self.parameters["detailed_test"]:
            j = 0
            errors_corr = pd.DataFrame(columns=self.met_name, index=["damping", 'frequency', 'Phase', 'SNR'])
            for idx, name in enumerate(self.met_name):
                # model = LinearRegression().fit(ampl_t[:, j].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1)))
                # errors_averaged.iloc[j] = [r2_score(ampl_t[:, idx], y_out[:, idx]),
                #                            mean_absolute_error(ampl_t[:, idx], y_out[:, idx]),
                #                            mean_squared_error(ampl_t[:, idx], y_out[:, idx]),
                #                            mean_absolute_percentage_error(ampl_t[:, idx], y_out[:, idx]) * 100,
                #                            model.score(ampl_t[:, j].reshape((-1, 1)), y_out[:, idx].reshape((-1, 1))),
                #                            model.intercept_,
                #                            model.coef_
                #                            ]
                #
                # yerr = 100 * np.abs(np.sqrt(y_out_var[:, idx]) / y_out[:, idx])
                yerr_ale = 100 * np.abs(np.sqrt(aleatoric_unc[:, idx]) / y_out[:, idx])
                yerr_epis = 100 * np.abs(np.sqrt(epistemic_unc[:, idx]) / y_out[:, idx])
                # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], c=yerr, c_map=cmap)
                # plt.title(self.met_name[idx])
                # self.savefig(id + "seperated_percent" + name)
                #
                # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], c=yerr_ale, c_map='Reds')
                # plt.title(self.met_name[idx])
                # self.savefig(id + "seperated_percent_ale" + name)
                #
                # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], c=yerr_epis, c_map='Greens')
                # plt.title(self.met_name[idx])
                # self.savefig(id + "seperated_percent_epis" + name)

                # yerr = np.abs(np.sqrt(y_out_var[:, idx]))
                # yerr_ale = np.abs((aleatoric_unc[:, idx]))
                # yerr_epis = np.abs((epistemic_unc[:, idx]))
                # self.modified_bland_altman_plot(100*ampl_t[:, idx]/ampl_t[:, idx], 100*y_out[:, idx]/ampl_t[:, idx], c=yerr, c_map=cmap)
                # plt.title(self.met_name[idx] + "corr: {}".format(np.corrcoef((ampl_t[:, idx]-y_out[:, idx]),yerr)))
                # self.savefig(id + "seperated" + name)

                self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], c=yerr_ale, c_map='Reds')
                plt.title(self.met_name[idx] + "corr: {}".format(np.corrcoef((ampl_t[:, idx]-y_out[:, idx]),yerr_ale)))
                self.savefig(id + "seperated_ale" + name)

                self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], c=yerr_epis, c_map='Greens')
                plt.title(self.met_name[idx] + "corr: {}".format(np.corrcoef((ampl_t[:, idx]-y_out[:, idx]),yerr_epis)[0,1]))
                self.savefig(id + "seperated_epis" + name)

                j += 1

            for idx_null, name in enumerate(selected_met):
                idx = self.met_name.index(name)
                # err = np.abs(y_out[:, idx] - ampl_t[:, idx])
                # yerr = 100 * np.abs(np.sqrt(y_out_var[:, idx]) / y_out[:, idx])
                yerr_ale = 100 * np.abs(np.sqrt(aleatoric_unc[:, idx]) / y_out[:, idx])
                yerr_epis = 100 * np.abs(np.sqrt(epistemic_unc[:, idx]) / y_out[:, idx])
                #
                # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=snrs, c=yerr, c_map=cmap)
                # plt.title(name)
                # self.savefig(id + "corrollation_precent" + 'snrs_' + name)
                #
                #
                # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=snrs, c=yerr_ale, c_map='Reds')
                # plt.title(name)
                # self.savefig(id + "corrollation_yerr_ale_precent" + 'snrs_' + name)
                #
                # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=snrs, c=yerr_epis, c_map='Greens')
                # plt.title(name)
                # self.savefig(id + "corrollation_yerr_epis_precent" + 'snrs_' + name)

                # yerr = np.abs(np.sqrt(y_out_var[:, idx]))
                yerr_ale = np.abs((aleatoric_unc[:, idx]))
                yerr_epis = np.abs((epistemic_unc[:, idx]))

                # self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=snrs, c=yerr, c_map=cmap)
                # plt.title(name)
                # self.savefig(id + "corrollation_" + 'snrs_' + name)


                self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=snrs, c=yerr_ale, c_map='Reds')
                plt.title(name)
                self.savefig(id + "corrollation_yerr_ale_" + 'snrs_' + name)

                plt.scatter(fr,yerr_ale)
                self.savefig(id + "yerr_ale_vs_" + 'fr' + name)

                self.modified_bland_altman_plot(ampl_t[:, idx], y_out[:, idx], gt=snrs, c=yerr_epis, c_map='Greens')
                plt.title(name)
                self.savefig(id + "corrollation_yerr_epis_" + 'snrs_' + name)

                j += 1
            errors_averaged.to_csv(self.saving_dir + id + "_errors_averaged.csv")


    # %%
    def test_asig(self,shift_t, alpha_t, ph_t, nl):
        sns.set_style('white')
        id = "test_" + str(shift_t) + "_" + str(alpha_t) + "_" + str(nl)
        ampl_t = self.min_c + (self.max_c - self.min_c)/2 + np.multiply(np.random.random(size=(1, 1+self.numOfSig)), (self.max_c - self.max_c))
        y_n, y_wn = self.getSignal(ampl_t, shift_t, alpha_t, ph_t, nl,True)

        y_test_np = y_n.astype('complex64')
        y_test = torch.from_numpy(y_test_np[:, 0:self.truncSigLen])
        print(y_test.size())
        ampl, shift, damp, ph, y_out, _ = self.predict_ensembles(y_test)
        y_out_f = fft.fftshift(fft.fft(y_out, axis=2))
        y_out_mean = np.mean(y_out_f, 0).T
        y_n, y_wn, y_out_mean,y_out_f = y_n/50, y_wn/50, y_out_mean/50,y_out_f/50
        self.plotppm(fft.fftshift(fft.fft((y_n[:, 0]), axis=0)), 0, 5, False, linewidth=1, linestyle='-')
        self.plotppm(y_out_mean, 0, 5, False, linewidth=1, linestyle='--')
        self.plotppm(35 + (fft.fftshift(fft.fft((y_n[:, 0]), axis=0)) - np.squeeze(y_out_mean)), 0, 5, False, linewidth=1,
                linestyle='-')
        self.plotppm(37.5 +(fft.fftshift(fft.fft((y_wn[:, 0]), axis=0)) - np.squeeze(y_out_mean)), 0, 5, True, linewidth=1,
                linestyle='-')
        self.plot_basis(ampl/25, shift, damp,ph)
        # self.fillppm(30-2*np.std(y_out_f, 0).T, 30+2*np.std(y_out_f, 0).T, 0, 5, True, alpha=.1, color='red')
        y_f = fft.fftshift(fft.fft(y_n, axis=0), axes=0)
        plt.title(self.cal_snr(y_f))
        self.savefig(id + "_tstasig")
        
        # print(ampl_t - ampl)
        # print(np.sqrt(ampl_var))
        # print(self.cal_snrf(fft.fftshift(fft.fft(y_n))))


    # %%
    def testmodel(self, model, x,enable_dropout=False):
        model.eval()
        if enable_dropout==True:
            model.enable_dropout()
        with torch.no_grad():
            # temp = model.forward(x)
            # dec_real, enct, enc, fr, damp, ph, mm, dec, decoded, b_spline_rec,ph_sig,recons_f = model.forward(x)
            condition = torch.IntTensor([9]).to(x.device)
            dec_real, enct, enc, fr, damp, ph, mm, dec, decoded, b_spline_rec,spline_coeff = model.forward(x,condition)
            # mu = enct[:, 0:self.param.numOfSig]
            # logvar = enct[:, self.param.numOfSig:2*(self.param.numOfSig)]
            mu = enct
            logvar = 0
            _, recons_loss = [lo / len(x) for lo in
                                     model.loss_function(dec, x, mu, logvar, mm, decoded, 0, enc, b_spline_rec,spline_coeff,condition)]
            # _, recons_loss = [lo / len(x) for lo in
            #                          model.loss_function(dec, x, mu, logvar, mm, decoded, 0, enc, b_spline_rec,ph_sig,recons_f)]
        return dec_real, enct, enc, fr, damp, ph, mm, dec, decoded, b_spline_rec,recons_loss

    # %%
    def getCRLB(self, model, x, ampl = None):
        noise = torch.std(x.T.real[-(68 + 128):-(68+1)], axis=0)
        model.eval()
        if ampl == None:
            cal_met = True
        else:
            cal_met = False
        with torch.no_grad():
            temp = model.crlb(x,noise_sd=noise,ampl=ampl, percent=False,cal_met=cal_met)
        return temp
    def sigmoid(self,x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig
    def predict_ensembles(self,y_test, mean_= True):
        decs = []
        encts = []
        encs = []
        frl = []
        dampl = []
        phl = []
        decodedl = []
        mml = []
        ph_sigs = []
        splines = []
        sp_torch = torch.nn.Softplus()
        def sp(x):
            return np.log(1+np.exp(x))
        def sp_inv(x):
            return np.log(np.exp(x)-1)
        for autoencoder in self.autoencoders:
            dect, enct, enc, fr, damp, ph, mm_rec, dec, decoded, spline, loss= self.testmodel(autoencoder,y_test.to(self.parameters["gpu"]))
            decs.append(dect.cpu().detach().numpy())
            mml.append(mm_rec.cpu().detach().numpy())
            encts.append((enct).cpu().detach().numpy())
            encs.append((enc).cpu().detach().numpy())
            frl.append(fr.cpu().detach().numpy())
            dampl.append(damp.cpu().detach().numpy())
            phl.append(ph.cpu().detach().numpy())
            splines.append(spline.cpu().detach().numpy())
            decodedl.append(decoded.cpu().detach().numpy())
            # ph_sigs.append(ph_sig.cpu().detach().numpy())
        if mean_:
            shift = (sum(frl) / len(frl))
            damp = (sum(dampl) / len(dampl))
            ph = (sum(phl) / len(phl))
        else:
            shift = np.asarray(frl)
            damp = np.asarray(dampl)
            ph = np.asarray(phl)
        encts_np = np.asarray((encts[0:]))
        if self.MM_type == 'single' or self.MM_type == 'single_param':
            mean = np.concatenate((encts_np[:, :, 0:self.numOfSig],np.expand_dims(encts_np[:, :, 2*(self.numOfSig)],axis=2)), axis=2)
            logvar = np.concatenate(
                (encts_np[:, :, self.numOfSig:2 * self.numOfSig], np.expand_dims(encts_np[:, :, 2*(self.numOfSig)+1], axis=2)), axis=2)
            ampl = (np.mean(mean, 0))
            std = np.exp(0.5 * logvar)
            aleatoric_unc = np.mean((std**2),0)
            epistemic_unc =  np.mean((mean ** 2),0) - (ampl ** 2)
            ampl = sp(ampl)
            ampl_var = aleatoric_unc + epistemic_unc
        else:
            # ampl = encs_np[:, :, 0:self.numOfSig]
            # if mean_:
            #     ampl = (np.mean(encs_np[:, :, 0:self.numOfSig], 0))
            # size_ = int(encts_np.shape[2]/2)
            mean = encts_np[:, :, :]
            mean = mean.astype(dtype=np.float128)[:,:,0:self.numOfSig]
            mean=sp(mean.astype(dtype=np.float128))
            # logvar = encts_np[:, :, size_:2 * size_]
            ampl = (np.mean((mean), 0))
            # epistemic_unc =  np.sqrt(sp(np.mean((mean ** 2),0) - (ampl ** 2)))
            epistemic_unc = ((np.var(mean ,0)))
        mean_ = encts_np.mean(0)
        return ampl, mean_, shift, damp, ph, np.asarray(decs), encts_np[:, :, 0:self.numOfSig],epistemic_unc, 0, np.asarray(decodedl),np.asarray(mml),np.asarray(splines),loss#,np.asarray(ph_sigs)

    def predict_MCDO(self,y_test, forward_passes=10, mean_= True):
        decs = []
        encts = []
        encs = []
        frl = []
        dampl = []
        phl = []
        decodedl = []
        mml = []
        sp_torch = torch.nn.Softplus()
        autoencoder = self.autoencoders[0]

        def sp(x):
            return np.log(1+np.exp(x))

        for i in range(forward_passes):
            # expect dect, enct, enc, fr, damp, ph,mm_rec,dec

            dect, enct, enc, fr, damp, ph, mm_rec, dec, decoded = self.testmodel(autoencoder,y_test.T.to(self.parameters["gpu"]),enable_dropout=True)
            decs.append(dect.cpu().detach().numpy())
            mml.append(mm_rec.cpu().detach().numpy())
            encts.append((enct).cpu().detach().numpy())
            encs.append((enc).cpu().detach().numpy())
            frl.append(fr.cpu().detach().numpy())
            dampl.append(damp.cpu().detach().numpy())
            phl.append(ph.cpu().detach().numpy())
            decodedl.append(decoded.cpu().detach().numpy())

        if mean_:
            shift = (sum(frl) / len(frl))
            damp = (sum(dampl) / len(dampl))
            ph = (sum(phl) / len(phl))
        else:
            shift = np.asarray(frl)
            damp = np.asarray(dampl)
            ph = np.asarray(phl)
        encts_np = np.asarray((encts))
        if self.MM_type == 'single' or self.MM_type == 'single_param':
            mean = np.concatenate((encts_np[:, :, 0:self.numOfSig],np.expand_dims(encts_np[:, :, 2*(self.numOfSig)],axis=2)), axis=2)
            logvar = np.concatenate(
                (encts_np[:, :, self.numOfSig:2 * self.numOfSig], np.expand_dims(encts_np[:, :, 2*(self.numOfSig)+1], axis=2)), axis=2)
            ampl = (np.mean(mean, 0))
            std = np.exp(0.5 * logvar)
            aleatoric_unc = sp(np.mean((std**2),0))
            epistemic_unc =  np.mean((mean ** 2),0) - (ampl ** 2)
            ampl = sp(ampl)
            ampl_var = aleatoric_unc + epistemic_unc

        else:
            # ampl = encs_np[:, :, 0:self.numOfSig]
            # if mean_:
            #     ampl = (np.mean(encs_np[:, :, 0:self.numOfSig], 0))
            size_ = int(encts_np.shape[2]/2)
            mean = encts_np[:, :, 0:size_]
            mean = mean.astype(dtype=np.float128)[:,:,0:self.numOfSig]
            mean.astype(dtype=np.float128)
            logvar = encts_np[:, :, size_:2 * size_]
            ampl = (np.mean(mean, 0))
            std = np.exp(0.5 * logvar)[:,:,0:self.numOfSig]
            aleatoric_unc = sp(np.sqrt(np.mean((std**2),0))*ampl)
            # epistemic_unc =  np.sqrt(sp(np.mean((mean ** 2),0) - (ampl ** 2)))
            epistemic_unc = sp(np.sqrt(np.var(mean,0)))
            ampl = sp(ampl)
            ampl_var = aleatoric_unc + epistemic_unc
        # ampl = np.mean(np.asarray(encs),0)
        return ampl, ampl_var, shift, damp, ph, np.asarray(decs), encts_np[:, :, 0:self.numOfSig],epistemic_unc, aleatoric_unc, np.asarray(decodedl),np.asarray(mml)

    def quantify_whole_subject(self, crlb=False):
        sns.set_style('white')
        path = "whole_Subj_l2/"
        Path(self.saving_dir+path).mkdir(exist_ok=True)
        data= np.load("data/8 subj/MRSI_8volunteers_raw_data/data_HC_from_eva/test_data.npy")
        mask = np.load("data/8 subj/MRSI_8volunteers_raw_data/data_HC_from_eva/mask_data.npy")
        mask = mask.squeeze()
        y_n= data.squeeze()
        nx, ny, nz, nt = y_n.shape
        y_n = y_n[mask.astype(bool),:]


        y_test_np = y_n.astype('complex64')

        if self.data_conj == True:
            y_test_np = np.conj(y_test_np)
        else:
            y_test_np = y_test_np
        # y = y[4:,:]
        y_test_np = self.normalize(y_test_np.T)
        ang = np.angle(y_test_np[1, :])
        y_test_np = y_test_np * np.exp(-1 * ang * 1j)
        y_test = torch.from_numpy(y_test_np.T[:,0:self.org_truncSigLen])
        print(y_test.size())
        ampl, mean_, shift, damp, ph, decs, encs, epistemic_unc, aleatoric_unc, decoded_net, mm_,spline,loss = self.predict_ensembles(y_test,mean_=False)
        with open(self.saving_dir + path +'loss.txt','w') as f:
            f.write(str(loss.cpu().numpy()))
            f.close()
        ampl_resh = np.zeros(((nx,ny,nz,ampl.shape[-1])))
        ampl_resh[mask.astype(bool),:] = ampl

        fig, axs = plt.subplots(2,4, figsize=(8,4))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(ampl_resh[:, :, 16, i].T)
            ax.set_title(self.met_name[i])
            ax.axis('off')
        self.savefig(path +"met_map")
        for i, name in enumerate(self.met_name):
        # Create a NIfTI image from the data array
            nifti_image = nib.Nifti1Image(np.expand_dims(ampl_resh[:,:,:,i]/ampl_resh[:,:,:,0],3), affine=np.eye(4))
            output_file = self.saving_dir+path + f"{name}.nii.gz"  # Replace with your desired output file path
            nib.save(nifti_image, output_file)

        y_out_f = fft.fftshift(fft.fft(decs[:,:,0:self.org_truncSigLen], axis=2),axes=2)
        sns.set_palette('Set2')
        sns.set_style('white')

        snr = self.cal_snrf(fft.fftshift(fft.fft(y_test_np.T,axis=1),axes=1),range=[2.5,3.5])
        rang = [1, 7]
        indic = list(range(len(y_test[:, 0])))
        indic.sort(key=lambda x: snr[x], reverse=False)
        for ll in range(0,12):
            id = indic[ll]
            sd_f = 5
            self.plotppm(np.fft.fftshift(np.fft.fft((y_test[ id,0:2048])).T), rang[0], rang[1], False,
                         linewidth=0.3, linestyle='-',label='signal')
            self.plotppm(y_out_f[0,id, :], rang[0],
                rang[1], False, linewidth=1, linestyle='-', label='Fit')
            if self.ens>1:
                self.fillppm(30+np.expand_dims(y_out_f[0,id, :] - sd_f * np.std(y_out_f[1:,id, :], 0),axis=1),
                             30+np.expand_dims(y_out_f[0,id, :] + sd_f * np.std(y_out_f[1:,id, :], 0),axis=1), 0, 5, False, alpha=.1,color='blue')

            self.plotppm(-1+
                np.fft.fftshift(np.fft.fft((y_test[id,0:self.org_truncSigLen]))) - y_out_f[0,id, 0:self.org_truncSigLen],
                         rang[0], rang[1],
                         True, linewidth=1, linestyle='-',label='residual')

            sns.despine()

            self.savefig(path +str(ll)+"_tstasig")

            # self.plot_basis(10*np.expand_dims(ampl_resh[id, :],axis=0), shift[id, :],
            #                 damp[id, :],
            #                 ph[id, :], rng=[1,8])
            # self.savefig(path + str(ll) + "basis")

            if self.parameters['spline']:
                self.p1 = int(self.ppm2p(self.parameters['fbound'][2], (self.truncSigLen)))
                self.p2 = int(self.ppm2p(self.parameters['fbound'][1], (self.truncSigLen)))
                self.in_size = int(self.p2-self.p1)
                y_test_ = (y_test[id, :]).unsqueeze(0)
                y_test_ = self.zero_fill_torch(y_test_, 1, self.parameters['zero_fill'][1]).cpu().detach()
                plt.plot(torch.fft.fftshift(torch.fft.fft((y_test_)).T)[self.p1:self.p2])
                rec_sig = self.zero_fill_torch(torch.from_numpy(decs[0, id]).unsqueeze(0), 1,
                                               self.parameters['zero_fill'][1]).cpu().detach()
                plt.plot(torch.fft.fftshift(torch.fft.fft((rec_sig)).T)[self.p1:self.p2] + spline[:,id].T)
                plt.plot(3+torch.fft.fftshift(torch.fft.fft((rec_sig-y_test_)).T)[self.p1:self.p2] + spline[:,id].T)
                plt.plot(spline[0,id])
                self.savefig(path + str(ll) + "result")

            rec_sig = self.zero_fill_torch(torch.from_numpy(decs[0, :]), 1,
                                               self.parameters['zero_fill'][1]).cpu().detach()
            y_test_ = self.zero_fill_torch(y_test, 1, self.parameters['zero_fill'][1]).cpu().detach()

            y_pred_trunc_f = np.zeros((nx,ny,nz,self.p2-self.p1))
            y_true_trunc_f = np.zeros((nx, ny, nz, self.p2 - self.p1))
            splines = np.zeros((nx, ny, nz, self.p2 - self.p1))
            y_pred_trunc_f[mask.astype(bool),:] = torch.fft.fftshift(torch.fft.fft(rec_sig,dim=1),dim=1)[:,self.p1:self.p2] + spline[0, :, :]
            y_true_trunc_f[mask.astype(bool),:] = torch.fft.fftshift(torch.fft.fft(y_test_,dim=1),dim=1)[:,self.p1:self.p2]
            splines[mask.astype(bool),:] = spline

            sio.savemat(self.saving_dir + path + 'rslt', {
                "y_pred_trunc_f" : y_pred_trunc_f,
                "y_true_trunc_f" : y_true_trunc_f,
                "spline" : splines
            })

    def quantify(self, crlb=False):
        sns.set_style('white')
        path = "rslt_wiithout_ph_1/"
        Path(self.saving_dir+path).mkdir(exist_ok=True)
        data= np.load(self.saving_dir + "test_" + str(self.test_nos)+ ".npz")
        _, y_n, snr, idx = [data[x] for x in data]
        y_test_np = y_n.astype('complex64')
        y_test = torch.from_numpy(y_test_np.T[:, 0:self.org_truncSigLen])
        print(y_test.size())
        ampl, mean_, shift, damp, ph, decs, encs, epistemic_unc, aleatoric_unc, decoded_net, mm_,spline,loss = self.predict_ensembles(y_test,mean_=False)

        with open(self.saving_dir + path +'loss.txt','w') as f:
            f.write(str(loss.cpu().numpy()))
            f.close()

        autoencoder=self.autoencoders[0]
        if crlb == True:
            crlb_ = []
            for i , j in zip(y_test[:,:].T,mean_):
                crlb_.append(self.getCRLB(autoencoder, i.T.to(self.parameters["gpu"]), ampl=torch.from_numpy(j).to(self.parameters["gpu"]).unsqueeze(0)).cpu().detach().numpy())
            aleatoric_unc=(np.asarray(crlb_)[:, 0:self.numOfSig])
        else:
            aleatoric_unc = epistemic_unc
        shift = np.mean(shift,0)
        damp = np.mean(damp, 0)
        ph = np.mean(ph,0)

        # decoded_net = decoded_net[:, :, 0, :1024] + 1j * decoded_net[:, :, 1, :1024]
        # decoded_net = self.zero_fill(decoded_net[:,:,0,:1024] + 1j*decoded_net[:,:,1,:1024],2,2048)
        # decoded_net_f = fft.fftshift(fft.fft(decoded_net, axis=2), axes=2)
        if self.MM:
            mm_f = fft.fftshift(fft.fft(mm_[:,:,0:2048], axis=2), axes=2)
        # ampl, shift, damp, ph, y_out, _ = self.predict_ensembles(y_test,mean_=False)
        y_out_f = fft.fftshift(fft.fft(decs[:,:,0:self.org_truncSigLen], axis=2),axes=2)
        # y_out_mean = np.mean(y_out_f, 0).T
        # y_n, y_wn, y_out_mean,y_out_f = y_n/50, y_wn/50, y_out_mean/50,y_out_f/50
        sns.set_palette('Set2')
        sns.set_style('white')
        # for ll in range(0,5):
        #     rng = range(ll*60,(ll+1)*64)
        #     self.plotsppm(fft.fftshift(fft.fft((y_n[0:1024, rng]), axis=0)), 0, 5, False, linewidth=0.3, linestyle='-')
        #     self.plotsppm(-30 + y_out_f[0, rng, 0:1024].T, 0, 5, False, linewidth=0.3, linestyle='--')
        #     self.plotsppm(+30 + (fft.fftshift(fft.fft((y_n[0:1024, rng]), axis=0)) - np.squeeze(y_out_f[0, rng, :].T)), 0, 5,
        #                   True,
        #                   linewidth=0.3,
        #                   linestyle='-')
        #     self.savefig(path + "subject_"+ str(ll) +"_tstasig")

        df_amplt = pd.DataFrame(ampl,columns=self.met_name)
        df_amplt['SNRs'] = snr
        df_amplt.to_csv(self.saving_dir + path + "result.csv")

        header_ale = ['Aleatoric Uncertainty ' + name for name in self.met_name]
        df_amplt = pd.DataFrame(aleatoric_unc,columns=header_ale)
        df_amplt.to_csv(self.saving_dir + path + "result_Ale.csv")

        header_ale = ['Epistemic Uncertainty ' + name for name in self.met_name]
        df_amplt = pd.DataFrame(epistemic_unc,columns=header_ale)
        df_amplt.to_csv(self.saving_dir + path + "result_Epis.csv")

        # file = open(self.saving_dir + path + '_predicts.csv', 'w')
        # writer = csv.writer(file)
        # writer.writerows(np.concatenate((ampl, y_out_var,epistemic_unc,aleatoric_unc), axis=1))
        # file.close()
        # self.plot_basis(ampl/25, shift, damp,ph)
        # self.fillppm(30-2*np.std(y_out_f, 0).T, 30+2*np.std(y_out_f, 0).T, 0, 5, True, alpha=.1, color='red')
        # y_f = fft.fftshift(fft.fft(y_n, axis=0), axes=0)
        # plt.title(self.cal_snr(y_f))
        snr = self.cal_snrf(fft.fftshift(fft.fft(y_test_np.T,axis=1),axes=1),range=[2.5,3.5])
        rang = [1, 7]
        indic = list(range(len(y_test[:, 0])))
        indic.sort(key=lambda x: snr[x], reverse=True)
        sio.savemat(self.saving_dir + path + "_testDB.mat",
                    {'y_test': y_test.numpy(), 'ampl': ampl, 'fit': decs, 'background': mm_})
        for ll in range(0,12):
            id = indic[ll]
            sd_f = 5
            self.plotppm(np.fft.fftshift(np.fft.fft((y_test[ id,0:2048])).T), rang[0], rang[1], False,
                         linewidth=0.3, linestyle='-',label='signal')
            # rec_signal, _, enc, fr, damp, ph, mm_v, _ = self.testmodel(self.autoencoders[0], y_test.T.cuda())
            if self.MM:
                self.plotppm(
                mm_f[0][id, :], rang[0],
                    rang[1], False, linewidth=1, linestyle='-',label='Background')
                if self.ens>1:
                    self.fillppm(15+np.expand_dims(mm_f[0,id, :] - sd_f * np.std(mm_f[1:,id, :], 0),axis=1),
                                 15+np.expand_dims(mm_f[0,id, :] + sd_f * np.std(mm_f[1:,id, :], 0),axis=1), 0, 5, False, alpha=.1,color='orange')

            self.plotppm(
            y_out_f[0,id, :], rang[0],
                rang[1], False, linewidth=1, linestyle='-', label='Fit')
            if self.ens>1:
                self.fillppm(30+np.expand_dims(y_out_f[0,id, :] - sd_f * np.std(y_out_f[1:,id, :], 0),axis=1),
                             30+np.expand_dims(y_out_f[0,id, :] + sd_f * np.std(y_out_f[1:,id, :], 0),axis=1), 0, 5, False, alpha=.1,color='blue')


            # self.plotppm(20+
            # decoded_net_f[0,id, :], rang[0],
            #     rang[1], False, linewidth=1, linestyle='-')
            # if self.ens > 1:
                # self.fillppm(20+np.expand_dims(decoded_net_f[0,id, :]- sd_f * np.std(decoded_net_f[1:,id, :], 0),axis=1),
                #              20+np.expand_dims(decoded_net_f[0,id, :] + sd_f * np.std(decoded_net_f[1:,id, :], 0),axis=1), 0, 5, False, alpha=.1,color='red')

            self.plotppm(-1+
                np.fft.fftshift(np.fft.fft((y_test[id,0:self.org_truncSigLen]))) - y_out_f[0,id, 0:self.org_truncSigLen],
                         rang[0], rang[1],
                         True, linewidth=1, linestyle='-',label='residual')

            # self.plotppm(5 +
            #     np.fft.fftshift(np.fft.fft((y_test[ id,0:2048]))) - decoded_net_f[0,id, :],
            #              rang[0], rang[1],
            #              True, linewidth=1, linestyle='-')

            sns.despine()
            # plt.legend(['y_test', 'fit','decoded_net','y_test-fit','y_test-decoded_net'])


            self.savefig(path +str(ll)+"_tstasig")

            self.plot_basis(10*np.expand_dims(ampl[id, :],axis=0), shift[id, :],
                            damp[id, :],
                            ph[id, :], rng=[1,8])
            self.savefig(path + str(ll) + "basis")

            if self.parameters['spline']:
                self.p1 = int(self.ppm2p(self.parameters['fbound'][2], (self.truncSigLen)))
                self.p2 = int(self.ppm2p(self.parameters['fbound'][1], (self.truncSigLen)))
                self.in_size = int(self.p2-self.p1)
                y_test_ = (y_test[id, :]).unsqueeze(0)
                y_test_ = self.zero_fill_torch(y_test_, 1, self.parameters['zero_fill'][1]).cpu().detach()
                plt.plot(torch.fft.fftshift(torch.fft.fft((y_test_)).T)[self.p1:self.p2])
                rec_sig = self.zero_fill_torch(torch.from_numpy(decs[0, id]).unsqueeze(0), 1,
                                               self.parameters['zero_fill'][1]).cpu().detach()
                plt.plot(torch.fft.fftshift(torch.fft.fft((rec_sig)).T)[self.p1:self.p2] + spline[:,id].T)
                plt.plot(3+torch.fft.fftshift(torch.fft.fft((rec_sig-y_test_)).T)[self.p1:self.p2] + spline[:,id].T)
                plt.plot(spline[0,id])
                self.savefig(path + str(ll) + "result")


    def dotrain(self,enc_num_manual=0):
        if self.MM_plot == True:
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
                                                 1.2322E8), self.mm, self.saving_dir +'_mm.txt')
        if self.pre_plot == True:
            self.plot_basis2(self.basisset, 2)
        if self.tr is True:
            self.data_prep()
            autoencoders = []
            self.tic()
            self.parameters['gpu'] = "cuda:1"
            enc_list = [enc_num_manual]
            if self.parameters['gpu'] == 'cuda:0':
                gpu = [0]
            if self.parameters['gpu'] == 'cuda:1':
                gpu = [1]
            if self.parameters['gpu'] == 'cuda:2':
                gpu = [2]

            # pl.seed_everything(42)
            for i in enc_list:
                if i==0:
                    dataloader_train = DataLoader(self.train, batch_size=self.batchsize, shuffle=True, pin_memory=True,
                               num_workers=self.num_of_workers)
                    dataloader_test = DataLoader(self.val, batch_size=self.batchsize, pin_memory=True,
                               num_workers=self.num_of_workers)

                else:
                    # self.train, _ = random_split(self.train, [int(len(self.train) / (self.ens - 1)),
                    #                                                 len(self.train) - int(
                    #                                                     len(self.train) / (self.ens - 1))])
                    dataloader_train = DataLoader(self.train, batch_size=self.batchsize, shuffle=True, pin_memory=True,
                               num_workers=self.num_of_workers)
                    dataloader_test = DataLoader(self.val, batch_size=self.batchsize, pin_memory=True,
                               num_workers=self.num_of_workers)
                    # self.beta_step /= (self.ens - 1)
                    # self.max_epoch = int(self.max_epoch * (self.ens - 1))

                logger = TensorBoardLogger('tb-logs', name=self.loging_dir)
                lr_monitor = LearningRateMonitor(logging_interval='step')
                if self.parameters['early_stop'][0]:
                    early_stopping = EarlyStopping('val_recons_loss',patience=self.parameters['early_stop'][1])
                    trainer= pl.Trainer(max_epochs=self.max_epoch, logger=logger,callbacks=[early_stopping,lr_monitor],accelerator='gpu',devices=gpu)
                else:
                    trainer = pl.Trainer(max_epochs=self.max_epoch, logger=logger,callbacks=[lr_monitor],accelerator='gpu',devices=gpu)
                # trainer= pl.Trainer(gpus=1, max_epochs=self.max_epoch, logger=logger,callbacks=[lr_monitor])
                logger.save()
                device = torch.device(self.parameters['gpu'])
                temp = Encoder_Model(self.depths[i], self.betas[i],self.reg_wei[i],self).to(device)
                if self.parameters["transfer_model_dir"] is not None:
                    PATH = self.parameters["transfer_model_dir"] + "model_" + str(i) + ".pt"
                    temp.load_state_dict(torch.load(PATH, map_location=device))
                    # temp.cuda()
                try:
                    temp = temp.load_from_checkpoint(self.parameters["checkpoint_dir"],depth=self.depths[i], beta=self.betas[i],tr_wei=self.reg_wei[i],param=self).to(device)
                except:
                    print("check point couldn't be loaded")
                # x = summary(temp.met.to('cuda:0'), (2, 1024))

                # torch.set_num_threads(128)
                print(torch.get_num_threads())

                trainer.fit(temp, dataloader_train, dataloader_test)
                autoencoders.append(temp)
                PATH = self.saving_dir + "model_"+ str(i) + ".pt"
                # Save
                torch.save(temp.state_dict(), PATH)
                del temp
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.memory_summary(device=device, abbreviated=False)
            self.toc("trining_time")
    def dotest(self):
        print("evaluation")
        self.autoencoders = []
        for i in range(0, self.ens):
            device = torch.device(self.parameters['gpu'])
            # device = torch.device('cpu')
            model = Encoder_Model(self.depths[i], self.betas[i],self.reg_wei[i],self)
            PATH = self.saving_dir + "model_" + str(i) + ".pt"
            model.load_state_dict(torch.load(PATH, map_location=device))
            model.to(device=device)
            model.eval()
            self.autoencoders.append(model)
            # macs, params = profile(model.met, inputs=(torch.randn(1, 2, 1024).to('cuda:1'),))
            # # macs, params = get_model_complexity_info(model.met.encoder, (2, 1024), as_strings=True,
            # #                                          print_per_layer_stat=True, verbose=True)
            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            # x = summary(model.met,(2,1024))


        if self.parameters["simulated"] == False:
            # self.quantify(crlb=False)
            self.quantify_whole_subject()
        else:
            self.ensemble = True
            if self.ens == 1:
                self.ensemble = False
            self.test_compact(ensembles=self.ensemble,crlb=False, estimate=True)

            plt.close()
            # self.test(ensembles=self.ensemble,crlb=False)
            # plt.close()
            # self.test_MonteCarlo(ensembles=self.ensemble)
            # plt.close()
            # self.erroVsnoise(20, 1, 2, 5, 0,True)
            # plt.close()
            # self.test_asig(2, 5, 0, 0.5)
            # plt.close()



