"""
MRSI_fit - Training Engine
=========================

This module contains the main training engine for the MRSI (Magnetic Resonance Spectroscopic Imaging)
fitting model. The Engine class handles all aspects of the training and testing pipeline including:

- Data loading and preprocessing
- Model training with PyTorch Lightning
- Ensemble training and Monte Carlo Dropout
- Model evaluation and quantification
- CRLB (Cramér-Rao Lower Bound) calculation
- Results visualization and reporting

The engine supports various configurations for:
- Different metabolite basis sets
- Macromolecular modeling
- Data augmentation
- Quality filtering
- Multiple ensemble training

Author: [Your Name]
Date: [Date]
"""

import csv
import gc
import os
import time
import nibabel as nib
from pathlib import Path

import wandb
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import mat73
import pandas as pd
import scipy
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
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

from utils import Jmrui, watrem
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from Model import Encoder_Model
from utils.DataLoader_MRSI import MRSI_Dataset
from utils.utils import plot_MM, normalize, Gauss, Lornz, ppm2p, cal_snrf, plotppm, tic, zero_fill_torch, toc, cal_snr, \
    wighted_var, savefig, plot_basis, fillppm, safe_elementwise_division

fontsize = 16
wandb.login(key="6685dc9a345d67dad1e247ea589a63d4e669c6f4")

class Engine():
    """
    Main training engine for MRSI spectral fitting.
    
    This class orchestrates the entire training and testing pipeline for deep learning-based
    MRSI spectral fitting. It handles data preprocessing, model training, evaluation,
    and result generation.
    
    Attributes:
        parameters (dict): Configuration parameters for the experiment
        saving_dir (str): Directory for saving model checkpoints and results
        data_dir (str): Directory containing training data
        basis_dir (str): Directory containing metabolite basis sets
        basisset (np.ndarray): Loaded metabolite basis set
        met_name (list): List of metabolite names
        numOfSig (int): Number of metabolites in the basis set
        MM (bool): Whether to include macromolecular modeling
        MM_model (function): Function for macromolecular modeling (Lorentzian/Gaussian)
    """
    
    def __init__(self, parameters):
        """
        Initialize the training engine with experiment parameters.
        
        Args:
            parameters (dict): Dictionary containing all experiment configuration parameters
        """
        self.parameters = parameters

        # Set up directory structure for saving results
        self.saving_dir = parameters['parent_root'] + parameters['child_root'] + parameters['version']
        self.epoch_dir =  "epoch/"
        self.loging_dir = parameters['parent_root'] + parameters['child_root']
        self.data_dir = parameters['data_dir']
        self.data_dir_ny = parameters['data_dir_ny']
        self.basis_dir = parameters['basis_dir']
        self.test_data_root = parameters['test_data_root']
        Path(self.saving_dir).mkdir(parents=True, exist_ok=True)
        Path(self.saving_dir+self.epoch_dir).mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.max_epoch = parameters['max_epoch']
        self.batchsize = parameters['batchsize']
        self.numOfSample = parameters['numOfSample']
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

        # Load basis set configuration
        try:
            basis_name = parameters["basis_name"]
        except:
            basis_name = "data"

        try:
            self.num_of_workers = parameters["num_of_workers"]
        except:
            self.num_of_workers = 0

        # Load and process metabolite basis set
        if self.basis_dir is not None:
            self.basisset = (sio.loadmat(self.basis_dir).get(basis_name)).T
            try:
                if parameters['basis_conj']:
                    self.basisset = np.conj(self.basisset)
                if parameters['norm_basis']:
                    maxi = np.abs(self.basisset).max()
                    self.basisset = self.basisset/maxi
            except:
                print("couldn't read basisset")
        
        # Data processing parameters
        self.wr = parameters['wr']
        self.data_name = parameters['data_name']
        self.numOfSig = self.basisset.shape[1]
        print(self.numOfSig)
        self.sigLen = parameters['sigLen']
        self.truncSigLen = parameters['truncSigLen']
        self.org_truncSigLen = self.truncSigLen
        self.BW = 1 / self.t_step
        self.f = np.linspace(-self.BW / 2, self.BW / 2, self.sigLen)
        self.t = np.arange(0, self.sigLen) * self.t_step
        self.t = np.expand_dims(self.t, 1)
        
        # Macromolecular modeling parameters
        self.MM = parameters['MM']
        self.MM_f = parameters['MM_f']
        self.MM_d = np.array(parameters['MM_d'])
        self.MM_a = parameters['MM_a']
        self.MM_plot = parameters['MM_plot']
        self.pre_plot = parameters['pre_plot']
        self.basis_need_shift = parameters['basis_need_shift']
        self.aug_params = parameters['aug_params']
        self.tr_prc = parameters['tr_prc']
        self.in_shape= parameters['in_shape']
        self.enc_type = parameters['enc_type']
        self.banorm = parameters['banorm']
        self.getCRLB_vmap = torch.vmap(self.getCRLB,in_dims=(None,0,0))
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

        # Load macromolecular basis set if specified
        if self.MM_dir is not None:
            self.mm = sio.loadmat(self.MM_dir).get("MM")
            if parameters['norm_basis']:
                self.mm = (self.mm.T)/maxi
            if parameters['basis_conj']:
                self.mm = np.conj(self.mm)

        # Configure macromolecular modeling
        if self.MM:
            self.numOfMM = 0
            if self.MM_type == 'param' or self.MM_type =="combined":
                if parameters['MM_model'] == "lorntz":
                    self.MM_model = Lornz
                    self.MM_d = (np.pi * self.MM_d)

                if parameters['MM_model'] == "gauss":
                    self.MM_model = Gauss
                    self.MM_d = self.MM_d * self.trnfreq
                    self.MM_d = (np.pi * self.MM_d)/(2*np.sqrt(np.log(2)))
                self.numOfMM = len(self.MM_f)

            if self.MM_type == 'single' or self.MM_type == 'single_param' or self.MM_type =="combined":
                self.met_name.append("MM")
                self.numOfMM += 1
        else:
            self.numOfMM = 0

        # Visualization settings
        self.heatmap_cmap = sns.diverging_palette(20, 220, n=200)
        self.sim_now = parameters['sim_order'][0]
        self.sim_dir = parameters['sim_order'][1]

        # Additional parameters with defaults
        try:
            self.kw = self.parameters['kw']
        except:
            self.parameters['kw'] = 3
            self.kw = 3

        try:
            self.MM_conj = self.parameters['MM_conj']
        except:
            self.MM_conj = True
            print("take care MM is conjucated!")

        if self.MM_conj == False:
            self.MM_f = [zz - 4.7 for zz in  self.MM_f]

        # Apply frequency shift to basis set if needed
        if self.basis_need_shift[0] == True:
            self.basisset = self.basisset[:, :] * np.exp(
                2 * np.pi * self.ppm2f(self.basis_need_shift[1]) * 1j * self.t)

        # Configure frequency domain processing
        if self.parameters['domain'] == 'freq':
            self.p1 = int(ppm2p(self.trnfreq,self.t_step,self.parameters['fbound'][2], (self.truncSigLen)))
            self.p2 = int(ppm2p(self.trnfreq,self.t_step,self.parameters['fbound'][1], (self.truncSigLen)))
            self.in_size = int(self.p2-self.p1)

        if self.in_shape != 'real':
            self.parameters['numofsplines'] = self.parameters['numofsplines'] * 2

    def data_proc(self):
        """
        Process and prepare the dataset for training and testing.
        
        This method handles:
        - Water removal if specified
        - Data conjugation if needed
        - Data normalization and phase correction
        - Test set splitting and saving
        - SNR calculation for test data
        
        Returns:
            tuple: (training_data, test_data) - Processed training and test datasets
        """
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

        self.numOfSample = y.shape[1]
        y, _ = normalize(y)
        ang = np.angle(y[1, :])
        y = y * np.exp(-1 * (ang+np.pi) * 1j)
        try:
            reload = self.parameters['reload_test']
        except:
            reload = True
        test_data_path = os.path.join(self.saving_dir, "test_" + str(self.test_nos))

        if os.path.exists(test_data_path):
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
            snrs = cal_snr(self,fft.fftshift(fft.fft(y_test, axis=0), axes=0))
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

        return y,y_test

    def data_prep(self):
        """
        Prepare the complete dataset for training.
        
        This method:
        - Loads data from various formats (MAT, NPY)
        - Processes the data through data_proc()
        - Converts data to PyTorch tensors
        - Creates training and validation datasets
        """
        if self.data_dir is not None:
            try:
                self.dataset = scipy.io.loadmat(self.data_dir).get(self.data_name).T
            except:
                try:
                    self.dataset = mat73.loadmat(self.data_dir).get(self.data_name).T
                except:
                    try:
                        self.dataset = np.load(self.data_dir).T
                    except:
                        print("couldn't read data")

        y, self.y_test = self.data_proc()

        y_f = fft.fftshift(fft.fft(y, axis=0),axes=0)

        if self.pre_plot ==True:
            plt.hist(cal_snrf(self, y_f))
            plt.show()
            plotppm(self, fft.fftshift(fft.fft((y[:, 2000]), axis=0)), 1, 8, True, linewidth=1, linestyle='-')
            plt.show()

        self.numOfSample = np.shape(y)[1];
        y_norm = y
        del y, y_f
        self.to_tensor(y_norm)
        del y_norm
        del self.dataset, self.y_trun


    def to_tensor(self,y_norm):
        """
        Convert normalized data to PyTorch tensors and create datasets.
        
        Args:
            y_norm: Normalized spectral data
        """
        y_trun = y_norm[0:self.truncSigLen, :].astype('complex64')
        self.y_trun = torch.from_numpy(y_trun[:, 0:self.numOfSample].T)
        if self.parameters["simulated"] is False:
            y_test = self.y_test[0:self.truncSigLen, :].astype('complex64')
            self.y_test_trun = torch.from_numpy(y_test[:, 0:self.numOfSample].T)
            # self.train = MRSI_Dataset(self.y_trun, engine=self)
            # self.val = MRSI_Dataset(self.y_test_trun, engine=self)
            self.train = TensorDataset(self.y_trun)
            self.val = TensorDataset(self.y_test_trun)
        else:
            self.y_test_trun = self.y_trun
            labels = torch.from_numpy(np.hstack((self.ampl_t,self.alpha)))
            labels = labels.type(torch.float32)
            my_dataset = TensorDataset(self.y_trun,labels)
            self.train, self.val = random_split(my_dataset, [int((self.numOfSample) * self.tr_prc), self.numOfSample - int((self.numOfSample) * self.tr_prc)])

    def inputSig(self,x,p1=None,p2=None):
        """
        Prepare input signal for the neural network.
        
        This method handles:
        - Zero-filling if specified
        - FFT transformation for frequency domain processing
        - Frequency range selection
        - Complex/real data formatting
        
        Args:
            x: Input spectral data
            p1: Start frequency index (optional)
            p2: End frequency index (optional)
            
        Returns:
            torch.Tensor: Formatted input signal for the neural network
        """
        if self.parameters['domain'] == 'freq':
            if self.parameters['zero_fill'][0] == True:
                x = zero_fill_torch(x,1,self.parameters['zero_fill'][1])
            x = torch.fft.fftshift(torch.fft.fft(x, dim=1), dim=1)
            if p1==None and p2==None:
                p1 = int(ppm2p(self.trnfreq,self.t_step,self.parameters['fbound'][2], (self.truncSigLen)))
                p2 = int(ppm2p(self.trnfreq,self.t_step,self.parameters['fbound'][1], (self.truncSigLen)))

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
        tic(self)

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
                crlb_ = self.getCRLB_vmap(autoencoder, y_test, mean_)
                # crlb_ = []
                # for i , j in zip(y_test[:,:].T,mean_):
                #     crlb_.append(self.getCRLB(autoencoder, i.T.to(self.parameters["gpu"]), ampl=torch.from_numpy(j).to(self.parameters["gpu"]).unsqueeze(0)).cpu().detach().numpy())
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
            cond_ = self.parameters["cond_test"]
            print(cond_)
            dec_real, enct, enc, fr, damp, ph, mm, dec, decoded, b_spline_rec, spline_coeff,b_spline_rec_im = model.forward(x,cond=cond_)
            mu = enct[:, 0:enct.shape[1]]
            logvar = enct[:, enct.shape[1]:]
            _, recons_loss = [lo / len(x) for lo in
                                     model.loss_function(dec, x, mu, logvar, mm, decoded, 0, enc, b_spline_rec,spline_coeff,cond_,damp,b_spline_rec_im)]
            # _, recons_loss = [lo / len(x) for lo in
            #                          model.loss_function(dec, x, mu, logvar, mm, decoded, 0, enc, b_spline_rec,ph_sig,recons_f)]
        return dec_real, enct, enc, fr, damp, ph, mm, dec, decoded, b_spline_rec, recons_loss
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
        losses = []
        crlbs = []
        sp_torch = torch.nn.Softplus()
        def sp(x):
            return np.log(1+np.exp(x))
        def sp_inv(x):
            return np.log(np.exp(x)-1)
        ens_total_weight = sum(self.parameters['ens_weights'])
        for indx, autoencoder in enumerate(self.autoencoders):
            # ens_weight = self.parameters['ens_weights'][indx]/ens_total_weight
            autoencoder.cond = torch.ones((y_test.shape[0],),dtype=torch.int) * 4
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
            losses.append(loss.cpu().detach().numpy())
            # ph_sigs.append(ph_sig.cpu().detach().numpy())
            crlbs.append(self.getCRLB_vmap(autoencoder, y_test.to(self.parameters["gpu"]),
                                           (enct[:,0:autoencoder.enc_out]).detach()))
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
            mean = np.concatenate((encts_np[:, :, 0:self.numOfSig],np.expand_dims(encts_np[:, :, self.numOfSig],axis=2)), axis=2)
            logvar=0
            if self.betas[0] !=0:
                logvar = np.concatenate(
                    (encts_np[:, :, autoencoder.enc_out:autoencoder.enc_out+self.numOfSig], np.expand_dims(encts_np[:, :, autoencoder.enc_out+self.numOfSig], axis=2)), axis=2)
        else:
            mean = encts_np[:, :, :]
            mean = mean.astype(dtype=np.float128)[:,:,0:self.numOfSig]
        return  mean, logvar, shift, damp, ph, np.asarray(decs), encts_np[:, :, 0:self.numOfSig], np.asarray(decodedl),np.asarray(mml),np.asarray(splines),np.array(losses),crlbs
    def process_ens(self, mean, logvar, shift, damp, ph, decs, encts_np, decodedl,mml,splines,losses,crlbs):
        # ampl = (np.mean(mean, 0))
        ampl = (np.average(mean, 0, weights=self.parameters['ens_weights'][0:self.ens]))
        var = np.exp(logvar)
        # aleatoric_unc = np.mean(var, 0)
        aleatoric_unc=0
        epistemic_unc = 0
        if self.ens !=1:
            aleatoric_unc = (np.average(var, 0, weights=self.parameters['ens_weights'][0:self.ens]))
            # epistemic_unc = np.average((mean ** 2), 0) - (ampl ** 2)
            # ampl = np.maximum(ampl,0)
            # ampl_var = aleatoric_unc + epistemic_unc
            epistemic_unc = wighted_var(mean,self.parameters['ens_weights'][0:self.ens])#(np.average((mean-ampl)**2, 0, weights=self.parameters['ens_weights']))
        # mean=sp(mean.astype(dtype=np.float128))
            # logvar = encts_np[:, :, size_:2 * size_]
        # ampl = (np.mean((mean), 0))
            # epistemic_unc =  np.sqrt(sp(np.mean((mean ** 2),0) - (ampl ** 2)))
        # epistemic_unc = ((np.var(mean ,0)))
        mean_ = encts_np.mean(0)
        crlbs = torch.stack(crlbs).cpu().detach().numpy()
        return ampl, mean_, shift, damp, ph, np.asarray(decs), encts_np[:, :, 0:self.numOfSig],epistemic_unc, aleatoric_unc, np.asarray(decodedl),np.asarray(mml),np.asarray(splines),np.array(losses),crlbs#,np.asarray(ph_sigs)

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

    def quantify_whole_subject(self, test_path, mask_path, affine_path,crlb=False,plot_spec=False):
        print(affine_path)

        sns.set_style('white')
        for ((test_id,mask_id),affine_id) in zip(zip(test_path,mask_path),affine_path):
            if affine_id is not None:
                nifti_templ = nib.load(affine_id)
                affine = nifti_templ.affine
            else:
                affine = np.eye(4)
            #self.parameters['test_subjs']:
            temp = 'version_p'
            path = f'test/{os.path.basename(test_id[:-4])}/{self.parameters["cond_test"]}/'
            Path(self.saving_dir+path).mkdir(exist_ok=True,parents=True)
            data= np.load(test_id)#f"data/8 subj/MS_Patients_3Dconcept/test_data_p{test_id}.npy")
            mask = np.load(mask_id)#f'{test_id})#f"data/8 subj/MS_Patients_3Dconcept/test_mask_p{test_id}.npy")
            # data= np.load(f"data/8 subj/MRSI_8volunteers_raw_data/data_HC_from_eva/test_data_{test_id}.npy")
            # mask = np.load(f"data/8 subj/MRSI_8volunteers_raw_data/data_HC_from_eva/test_mask_{test_id}.npy")
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
            y_test_np,_ = normalize(y_test_np.T)
            ang = np.angle(y_test_np[1, :])
            y_test_np = y_test_np * np.exp(-1 * (ang+np.pi) * 1j)
            self = tic(self)
            y_test = torch.from_numpy(y_test_np.T[:,0:self.org_truncSigLen])
            print(y_test.device)
            print(y_test.size())
            rslt = self.predict_ensembles(
                y_test, mean_=False)

            ampl, mean_, shift, damp, ph, decs, encs, epistemic_unc, aleatoric_unc, decoded_net, mm_,spline,loss,crlbs = self.process_ens(*rslt)
            toc(self,'test_time')

            with open(self.saving_dir + path + 'loss.txt', 'w') as f:
                f.write(str(loss))
                f.close()

            ampl_resh = np.zeros(((nx,ny,nz,ampl.shape[-1])))

            ampl_resh[mask.astype(bool),:] = ampl

            epistemic_unc_resh= 0
            aleatoric_unc_resh = 0
            if self.ens!=1:
                epistemic_unc_resh = np.zeros(((nx,ny,nz,epistemic_unc.shape[-1])))
                epistemic_unc_resh[mask.astype(bool),:] = epistemic_unc
                aleatoric_unc_resh = np.zeros(((nx,ny,nz,aleatoric_unc.shape[-1])))
                aleatoric_unc_resh[mask.astype(bool),:] = aleatoric_unc

            crlb_unc_resh = np.zeros(((nx,ny,nz,crlbs.shape[-1])))
            crlb_unc_resh[mask.astype(bool),:] = crlbs.mean(0)

            # fig, axs = plt.subplots(4,4, figsize=(8,8))
            id_div_cr = self.met_name.index('Cr')
            id_div_pcr = self.met_name.index('PCr')
            # for i, ax in enumerate(axs.flatten()):
            #     ratio = safe_elementwise_division(ampl_resh[:, :, 16, i].T , (ampl_resh[:, :, 16, id_div_cr].T))
            #     ratio[ratio>2]=0
            #     ax.imshow(ratio)
            #     ax.set_title(f'{self.met_name[i]}/[Cr+PCr]')
            #     ax.axis('off')
            # savefig(self,path +"met_map")

            for i, name in enumerate(self.met_name):
                # Create a NIfTI image from the data array
                # ratio = safe_elementwise_division(ampl_resh[:, :, :, i] , (ampl_resh[:, :, :, id_div_cr]+ampl_resh[:, :, :, id_div_pcr]))
                # ratio[ratio > 10] = 0
                nifti_image = nib.Nifti1Image(np.expand_dims(ampl_resh[:, :, :, i], 3),
                                              affine=affine)
                # if i==id_div_cr or i==id:
                #     nifti_image = nib.Nifti1Image(np.expand_dims(ampl_resh[:, :, :, i], 3),
                #                                   affine=affine)
                # else:
                #     nifti_image = nib.Nifti1Image(np.expand_dims(ratio,3), affine=affine)
                output_file = self.saving_dir+path + f"{name}.nii.gz"  # Replace with your desired output file path
                nib.save(nifti_image, output_file)

            for i, name in enumerate(self.met_name):
                # Create a NIfTI image from the data array
                ratio = safe_elementwise_division(ampl_resh[:, :, :, i] , (ampl_resh[:, :, :, id_div_cr]+ampl_resh[:, :, :, id_div_pcr]))
                # ratio[ratio > 10] = 0
                # nifti_image = nib.Nifti1Image(np.expand_dims(ampl_resh[:, :, :, i], 3),
                #                               affine=affine)
                # if i==id_div_cr or i==id:
                #     nifti_image = nib.Nifti1Image(np.expand_dims(ampl_resh[:, :, :, i], 3),
                #                                   affine=affine)
                # else:
                nifti_image = nib.Nifti1Image(np.expand_dims(ratio,3), affine=affine)
                output_file = self.saving_dir+path + f"{name}_tCr.nii.gz"  # Replace with your desired output file path
                nib.save(nifti_image, output_file)
            if self.ens !=1:
                for i, name in enumerate(self.met_name):
                # Create a NIfTI image from the data array
                    nifti_image = nib.Nifti1Image(np.expand_dims(np.sqrt(epistemic_unc_resh[:, :, :, i]) / ampl_resh[:, :, :, i], 3),
                                                  affine=affine)
                    output_file = self.saving_dir + path + f"{name}_epi_unc.nii.gz"  # Replace with your desired output file path
                    nib.save(nifti_image, output_file)
                for i, name in enumerate(self.met_name):
                # Create a NIfTI image from the data array
                    nifti_image = nib.Nifti1Image(np.expand_dims(np.sqrt(aleatoric_unc_resh[:, :, :, i]) / ampl_resh[:, :, :, i], 3),
                                                  affine=affine)
                    output_file = self.saving_dir + path + f"{name}_ale_unc.nii.gz"  # Replace with your desired output file path
                    nib.save(nifti_image, output_file)

            for i, name in enumerate(self.met_name):
            # Create a NIfTI image from the data array
                nifti_image = nib.Nifti1Image(np.expand_dims(100*(crlb_unc_resh[:, :, :, i]) / ampl_resh[:, :, :, i], 3),
                                              affine=affine)
                output_file = self.saving_dir + path + f"{name}_crlb_unc.nii.gz"  # Replace with your desired output file path
                nib.save(nifti_image, output_file)

            if plot_spec:
                sns.set_palette('Set2')
                sns.set_style('white')

                self.p1 = int(ppm2p(self.trnfreq,self.t_step,self.parameters['fbound'][2], (self.truncSigLen)))
                self.p2 = int(ppm2p(self.trnfreq,self.t_step,self.parameters['fbound'][1], (self.truncSigLen)))
                self.in_size = int(self.p2 - self.p1)

                rec_sig = zero_fill_torch(torch.from_numpy(decs), 2,
                                               self.parameters['zero_fill'][1]).cpu().detach()
                mm_ = zero_fill_torch(torch.from_numpy(mm_), 2,
                                               self.parameters['zero_fill'][1]).cpu().detach()
                y_test_ = zero_fill_torch(y_test, 1, self.parameters['zero_fill'][1]).cpu().detach()

                y_out_f = fft.fftshift(fft.fft(rec_sig, axis=2),axes=2)
                mm_f = fft.fftshift(fft.fft(mm_, axis=2), axes=2)
                # y_out_f_mean = y_out_f.mean(0,keepdims=True)
                y_out_f_mean = np.average(y_out_f,0, keepdims=True, weights= self.parameters['ens_weights'][0:self.ens])
                mm_f_mean = np.average(mm_f, 0, keepdims=True, weights=self.parameters['ens_weights'][0:self.ens])
                spline_mean = np.average(spline, 0, keepdims=False, weights=self.parameters['ens_weights'][0:self.ens])
                # y_out_f_sd = y_out_f.std(0, keepdims=True)
                y_out_f_sd = wighted_var(y_out_f,self.parameters['ens_weights'][0:self.ens],keepdims=True)**0.5
                spline_mean_sd = wighted_var(spline, self.parameters['ens_weights'][0:self.ens], keepdims=False)**0.5
                snr = cal_snrf(self.trnfreq,self.t_step,fft.fftshift(fft.fft(y_test_np.T,axis=1),axes=1),range=[2.5,3.5])
                rang = [1.8, 3.8]
                indic = list(range(len(y_test[:, 0])))
                indic.sort(key=lambda x: snr[x], reverse=False)
                # indic = np.random.randint(0,(y_test_np.shape[1]),20)
                indic = [12114,12000,7461,13609,12005,62,6042,12458,15732,14018,9965,8223,15184,5458,14558,11586,11119,15075,7228,16825]

                for ll in range(0,12):

                    id = indic[ll]
                    sd_f = 2
                    plotppm(np.fft.fftshift(np.fft.fft((y_test_[ id,:])).T)[int(ppm2p(self.trnfreq,self.t_step,5, (self.truncSigLen))):
                                                                                 int(ppm2p(self.trnfreq,self.t_step,1, (self.truncSigLen)))], 1, 5, True,
                                 linewidth=2, linestyle='-',label='Signal')
                    savefig(self,path +str(ll)+"_whole")
                    plotppm(np.fft.fftshift(np.fft.fft((y_test_[ id,:])).T)[self.p1:self.p2], rang[0], rang[1], False,
                                 linewidth=2, linestyle='-',label='Signal')
                    plotppm(y_out_f_mean[0,id, self.p1:self.p2]+spline_mean[id,:], rang[0],
                        rang[1], False, linewidth=2, linestyle='-', label='Fit')
                    plotppm(mm_f_mean[0,id, self.p1:self.p2], rang[0],
                        rang[1], False, linewidth=2, linestyle='-', label='MM')
                    plotppm(spline_mean[id,:], rang[0],
                        rang[1], False, linewidth=2, linestyle='-', label='Spline')

                    if self.ens>1:
                        fillppm(self,np.expand_dims(spline_mean[id,:]+y_out_f_mean[0,id, self.p1:self.p2] - sd_f * (y_out_f_sd[0,id, self.p1:self.p2]+spline_mean_sd[id,:]),axis=1),
                                     np.expand_dims(spline_mean[id,:]+y_out_f_mean[0,id, self.p1:self.p2] + sd_f * (y_out_f_sd[0,id, self.p1:self.p2]++spline_mean_sd[id,:]),axis=1),
                                                    rang[0],rang[1], False, alpha=.1,color='orange')
                    if self.ens>1:
                        fillppm(self,np.expand_dims(spline_mean[id,:] - sd_f * (spline_mean_sd[id,:]),axis=1),
                                     np.expand_dims(spline_mean[id,:]+ sd_f * (+spline_mean_sd[id,:]),axis=1),
                                                    rang[0],rang[1], False, alpha=.1,color='blue')
                    plotppm(
                        np.fft.fftshift(np.fft.fft((y_test_[id,:])))[self.p1:self.p2]
                        -spline_mean[id,:]  - y_out_f_mean[0,id, self.p1:self.p2],
                                 rang[0], rang[1],
                                 True, linewidth=1, linestyle='-',label='Residual')
                    sns.despine()

                    savefig(self,path +str(ll)+"_tstasig")

                    plot_basis(self.trnfreq,self.t_step, self.basisset, self.t, self.met_name,10*np.expand_dims(ampl[id, :],axis=0), shift[0,id, :],
                                    damp[0,id, :],
                                    ph[0,id, :], rng=[1,5])
                    savefig(self,path + str(ll) + "basis")

                    # if self.parameters['spline']:
                    #     y_test_ = (y_test[id, :]).unsqueeze(0)
                    #     y_test_ = self.zero_fill_torch(y_test_, 1, self.parameters['zero_fill'][1]).cpu().detach()
                    #     plt.plot(torch.fft.fftshift(torch.fft.fft((y_test_)).T)[self.p1:self.p2])
                    #     rec_sig = self.zero_fill_torch(torch.from_numpy(decs.mean()[0, id]).unsqueeze(0), 1,
                    #                                    self.parameters['zero_fill'][1]).cpu().detach()
                    #     plt.plot(torch.fft.fftshift(torch.fft.fft((rec_sig)).T)[self.p1:self.p2] + spline[:,id].T)
                    #     plt.plot(1+torch.fft.fftshift(torch.fft.fft((rec_sig-y_test_)).T)[self.p1:self.p2] + spline[:,id].T)
                    #     plt.plot(spline[0,id])
                    #     self.savefig(path + str(ll) + "result")

                y_pred_trunc_f = np.zeros((self.ens,nx,ny,nz,self.in_size))
                y_true_trunc_f = np.zeros((nx, ny, nz, self.in_size))
                splines = np.zeros((self.ens,nx, ny, nz, self.in_size))
                y_pred_trunc_f[:,mask.astype(bool),:] = torch.fft.fftshift(torch.fft.fft(rec_sig,dim=2),dim=2)[:,:,self.p1:self.p2] + spline
                y_true_trunc_f[mask.astype(bool),:] = torch.fft.fftshift(torch.fft.fft(y_test_,dim=1),dim=1)[:,self.p1:self.p2]
                splines[:,mask.astype(bool),:] = spline

                sio.savemat(self.saving_dir + path + 'rslt_spectra.mat', {
                    "y_pred_trunc_f" : y_pred_trunc_f,
                    "y_true_trunc_f" : y_true_trunc_f,
                    "spline" : splines,
                })

                sio.savemat(self.saving_dir + path + 'rslt_map.mat', {
                    "mask" : mask,
                    "metabolite_map": ampl_resh,
                    "uncertainty_map": epistemic_unc_resh,
                    "metabolite_name": self.met_name
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
        self = tic(self)
        if self.MM_plot == True:
            plot_MM(self)
        self.data_prep()
        autoencoders = []
        # self.parameters['gpu'] = "cuda:0"
        enc_list = [enc_num_manual]
        if self.parameters['gpu'] == 'cuda:0':
            gpu = [0]
        if self.parameters['gpu'] == 'cuda:1':
            gpu = [1]
        if self.parameters['gpu'] == 'cuda:2':
            gpu = [2]
        print(f'the selected gpu is : {gpu}')

        # pl.seed_everything(42)
        for i in enc_list:
            self.epoch_dir = self.epoch_dir + str(i)+'/'
            Path(self.saving_dir + self.epoch_dir).mkdir(parents=True, exist_ok=True)
            device = torch.device(self.parameters['gpu'])
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

            # logger = TensorBoardLogger('tb-logs', name=self.loging_dir)
            logger = WandbLogger(project='dlfit')
            lr_monitor = LearningRateMonitor(logging_interval='step')
            if self.parameters['early_stop'][0]:
                early_stopping = EarlyStopping('val_recons_loss',patience=self.parameters['early_stop'][1])
                trainer= pl.Trainer(max_epochs=self.max_epoch, logger=logger,callbacks=[early_stopping,lr_monitor],accelerator='gpu',devices=gpu)
            else:
                trainer = pl.Trainer(max_epochs=self.max_epoch, logger=logger,callbacks=[lr_monitor],accelerator='gpu',devices=gpu,enable_checkpointing=True)
            # trainer= pl.Trainer(gpus=1, max_epochs=self.max_epoch, logger=logger,callbacks=[lr_monitor])
            logger.save()


            temp = Encoder_Model(self.depths[i], self.betas[i],self.reg_wei[i],i,self).to(device)
            if self.parameters["transfer_model_dir"] is not None:
                PATH = self.parameters["transfer_model_dir"] + "model_gpu_" + str(i) + ".pt"
                temp.load_state_dict(torch.load(PATH, map_location=device))
                # temp.cuda()
            try:
                temp = temp.load_from_checkpoint(self.parameters["checkpoint_dir"])#,depth=self.depths[i], beta=self.betas[i],tr_wei=self.reg_wei[i],param=self).to(device)
            except:
                print("check point couldn't be loaded")
            # x = summary(temp.met.to('cuda:0'), (2, 1024))

            torch.set_num_threads(16)
            print(torch.get_num_threads())

            trainer.fit(temp, dataloader_train, dataloader_test,ckpt_path=self.parameters["checkpoint_dir"])
            autoencoders.append(temp)
            PATH = self.saving_dir + "model_gpu_"+ str(i) + ".pt"
            # Save
            torch.save(temp.state_dict(), PATH)

            PATH = self.saving_dir + "model_cpu_"+ str(i) + ".pt"
            # Save
            torch.save(temp.to('cpu').state_dict(), PATH)
            del temp
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.memory_summary(device=device, abbreviated=False)
        toc(self,"trining_time")

    def inference(self, test_path,mask_path,affine_path):
        print("evaluation")
        self.autoencoders = []
        for i in range(0, self.ens):
            device = torch.device(self.parameters['gpu'])
            # device = torch.device('cpu')
            model = Encoder_Model(self.depths[i], self.betas[i],self.reg_wei[i],i,self).to(device)
            PATH = self.saving_dir + "model_cpu_" + str(i) + ".pt"
            model.load_state_dict(torch.load(PATH, map_location=device))
            model.to(device=device)
            model.eval()
            self.autoencoders.append(model)
        self.quantify_whole_subject(test_path,mask_path,affine_path, plot_spec=True)
    def dotest(self, test_path,mask_path,affine_path):
        print("evaluation")
        self.autoencoders = []
        for i in range(0, self.ens):
            device = torch.device(self.parameters['gpu'])
            # device = torch.device('cpu')
            model = Encoder_Model(self.depths[i], self.betas[i],self.reg_wei[i],i,self)
            PATH = self.saving_dir + "model_gpu_" + str(i) + ".pt"
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
            self.quantify_whole_subject(test_path,mask_path,affine_path,plot_spec=True)
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



