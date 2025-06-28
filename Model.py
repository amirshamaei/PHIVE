"""
MRSI_fit - Deep Learning Model for MRSI Spectral Fitting
=======================================================

This module contains the main neural network model for MRSI (Magnetic Resonance Spectroscopic Imaging)
spectral fitting. The Encoder_Model class implements a variational autoencoder-based approach for
quantifying metabolite concentrations from MRSI spectra.

Key Features:
- Variational Autoencoder (VAE) architecture with PyTorch Lightning
- Support for both convolutional and transformer encoders
- Macromolecular modeling (Lorentzian/Gaussian/Voigt)
- B-spline baseline modeling
- Ensemble training and Monte Carlo Dropout
- CRLB (Cramér-Rao Lower Bound) calculation
- Data augmentation during training

The model can handle:
- Complex and real-valued spectral data
- Frequency and time domain processing
- Multiple metabolite basis sets
- Quality filtering and uncertainty quantification

Author: [Your Name]
Date: [Date]
"""

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import numpy as np
import scipy.io as sio
import torchmetrics
from scipy.stats import pearsonr
from torch.func import jacfwd
from torchmetrics import R2Score, PearsonCorrCoef
from torch import Tensor
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt, colorbar
import nibabel as nib
from Models.UNET import ConvNet_ENC, ConvNet_DEC
import wandb


from torch.autograd.functional import jacobian
from Models.transformer import Transformer, TransformerB
from torchcubicspline import(natural_cubic_spline_coeffs,
                             NaturalCubicSpline)

from utils.utils import ppm2p, zero_fill_torch, plot_basis, plotppm, savefig, normalize, load_nifti_file


# from pytorch_memlab import LineProfiler, profile


class Encoder_Model(pl.LightningModule):
    """
    Variational Autoencoder model for MRSI spectral fitting.
    
    This model implements a deep learning approach to quantify metabolite concentrations
    from MRSI spectra. It uses a variational autoencoder architecture with:
    - An encoder that maps spectral data to latent parameters
    - A decoder that reconstructs spectra from metabolite parameters
    - Support for macromolecular modeling and baseline correction
    
    The model outputs:
    - Metabolite concentrations (amplitudes)
    - Frequency shifts
    - Damping factors
    - Phase corrections
    - Macromolecular contributions (if enabled)
    - B-spline baseline coefficients (if enabled)
    
    Attributes:
        param: Engine parameters containing configuration
        basis: Metabolite basis set
        met_name: List of metabolite names
        MM: Whether macromolecular modeling is enabled
        enc_type: Type of encoder ('conv' or 'trans')
        beta: Beta parameter for VAE regularization
    """

    def __init__(self,depth, beta, tr_wei, i,param):
        """
        Initialize the MRSI fitting model.
        
        Args:
            depth: Model depth parameter
            beta: Beta parameter for VAE regularization
            tr_wei: Training weight parameter
            i: Ensemble index
            param: Engine parameters containing all configuration
        """
        super().__init__()
        # self.save_hyperparameters()
        self.ens_indx = i
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.param = param
        self.met = []
        self.selected_met = ["Cr", "GPC", "sIns", "NAA", "PCho", "Tau"]
        
        # Register time vector and basis set as buffers
        self.register_buffer("t",torch.from_numpy(param.t).float()[0:self.param.org_truncSigLen])
        self.sw = 1/param.t_step
        self.register_buffer("basis",torch.from_numpy(param.basisset[:self.param.org_truncSigLen, 0:param.numOfSig].astype('complex64')))
        
        # Loss function and metrics
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.register_buffer("beta",nn.Parameter(torch.tensor(beta), requires_grad=False))
        self.r2 = torchmetrics.R2Score(adjusted=True)
        
        # T2 relaxation time parameter
        if self.param.MM_constr == True :
            print('tr is not in the model')
        else:
            self.register_buffer("tr",nn.Parameter(torch.tensor(0.004), requires_grad=True))
        self.tr_wei = tr_wei
        
        # Activation functions
        self.act = nn.Softplus()
        self.lact = nn.ReLU6()
        self.sigm = nn.Sigmoid()
        self.model = None
        self.tanh = nn.Tanh()
        
        # Configure output dimensions based on macromolecular modeling
        if self.param.MM == True :
            if self.param.MM_type == 'single' or self.param.MM_type == 'single_param':
                self.enc_out =  1 * (param.numOfSig) + 4 + 3 # numof metabolite + parameters + MM parameters
                self.register_buffer("mm",torch.from_numpy(param.mm[:self.param.org_truncSigLen,:].astype('complex64')).T)
            if self.param.MM_type == 'param':
                if self.param.MM_fd_constr == False:
                    self.enc_out =  (1* (self.param.numOfSig)+self.param.numOfMM*4 + 3 + (self.param.numOfMM))
                else:
                    self.enc_out =  1* (self.param.numOfSig)+self.param.numOfMM + 3 + 1 + 1 + 1
            if self.param.MM_type == 'combined':
                self.enc_out =  (1* (self.param.numOfSig)+self.param.numOfMM*4 + 3 + (self.param.numOfMM))
                self.register_buffer("mm",
                                     torch.from_numpy(param.mm[:self.param.org_truncSigLen, :].astype('complex64')).T)
        else:
            self.enc_out = 1 * (param.numOfSig) + 3

        if self.param.MM_constr == True:
            self.enc_out += 1

        # Add B-spline coefficients if enabled
        if self.param.parameters['spline']:
            self.enc_out = self.enc_out + self.param.parameters['numofsplines']

        # Dropout configuration
        try:
            self.dropout = param.parameters['dropout']
        except:
            self.dropout = 0

        # Input channel configuration
        if self.param.in_shape == 'real':
            self.in_chanel = 1
        else:
            self.in_chanel = 2

        # VAE output configuration
        if self.beta != 0:
            self.enc_out_ = 2 * self.enc_out
        else:
            self.enc_out_ = self.enc_out

        # Conditional training configuration
        self.cond_max = self.param.parameters["cond_max"]

        if self.cond_max != 1:
            self.bspline_embed = nn.ModuleList()
            for i in range(self.cond_max):
                self.bspline_embed.append(nn.Linear(12,(3*(i+1))))
            self.embed = (nn.Linear(1, 7*7))

        # Initialize encoder based on type
        if param.enc_type == 'conv':
            self.met = ConvNet_ENC(in_chanel=self.in_chanel,latent_Size=self.enc_out_, dropout=self.dropout,freeze_enc=self.param.parameters["freeze_enc"])
            if param.parameters['decode'] == True:
                self.decode = ConvNet_DEC(out_chanel=self.in_chanel,dropout=self.dropout, freeze_dec=self.param.parameters["freeze_dec"])
        if param.enc_type == 'trans':
            # self.met = Transformer(insize=self.in_size,outsize=self.enc_out_)
            self.met = TransformerB(in_channels=1, out_channels=512, num_heads=4, hidden_size=128, num_layers=2,outsize=self.enc_out_)
            if param.parameters['decode'] == True:
                self.decode = ConvNet_DEC(out_chanel=self.in_chanel, dropout=self.dropout,
                                          freeze_dec=self.param.parameters["freeze_dec"])
        
        # Macromolecular model configuration
        if param.parameters['MM_model'] == "lorntz":
            self.MM_model = self.Lornz
        if param.parameters['MM_model'] == "gauss":
            self.MM_model = self.Gauss

        # Metrics
        self.pearsoncorr = PearsonCorrCoef(num_outputs=self.param.numOfSig)
        
        # Zero-filling configuration
        if self.param.parameters['zero_fill'][0] == True:
            self.param.truncSigLen = self.param.parameters['zero_fill'][1]
        
        # Frequency domain configuration
        if self.param.parameters['domain'] == 'freq':
            self.p1 = int(ppm2p(self.param.trnfreq,self.param.t_step,self.param.parameters['fbound'][2], (self.param.truncSigLen)))
            self.p2 = int(ppm2p(self.param.trnfreq,self.param.t_step,self.param.parameters['fbound'][1], (self.param.truncSigLen)))
            self.in_size = int(self.p2-self.p1)
        
        # Vectorized augmentation function
        self.getaug_vmap = torch.vmap(self.get_augment, in_dims=(0, None, None, None, None),randomness='different')


    def sign(self,t,eps):
        """Sign function with epsilon smoothing to avoid division by zero."""
        return (t/torch.sqrt(t**2+ eps))

    def sigmoid(self,x,a, b):
        """Sigmoid function with parameters a and b."""
        return (1/(1+torch.exp(-1*a*(x-b))))
    def Gauss(self, ampl, f, d, ph, Crfr, Cra, Crd):
        """
        Gaussian macromolecular model.
        
        Args:
            ampl: Amplitude
            f: Frequency
            d: Damping
            ph: Phase
            Crfr: Frequency reference
            Cra: Amplitude reference
            Crd: Damping reference
            
        Returns:
            torch.Tensor: Gaussian macromolecular signal
        """
        return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                       torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                  torch.exp(-(d+Crd)**2 * self.t.T*self.t.T))
    def Lornz(self, ampl, f, d, ph, Crfr, Cra, Crd):
        """
        Lorentzian macromolecular model.
        
        Args:
            ampl: Amplitude
            f: Frequency
            d: Damping
            ph: Phase
            Crfr: Frequency reference
            Cra: Amplitude reference
            Crd: Damping reference
            
        Returns:
            torch.Tensor: Lorentzian macromolecular signal
        """
        return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                       torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                  torch.exp(-(d+Crd) * self.t.T))
    def Voigt(self, ampl, f, dl,dg, ph, Crfr, Cra, Crd):
        """
        Voigt macromolecular model (combination of Lorentzian and Gaussian).
        
        Args:
            ampl: Amplitude
            f: Frequency
            dl: Lorentzian damping
            dg: Gaussian damping
            ph: Phase
            Crfr: Frequency reference
            Cra: Amplitude reference
            Crd: Damping reference
            
        Returns:
            torch.Tensor: Voigt macromolecular signal
        """
        return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                       torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                  torch.exp(-(((dl) * self.t.T)+(dg+Crd) * self.t.T*self.t.T)))
    def model_decoder(self,enc,cond):
        """
        Decode the encoder output to reconstruct the spectral signal.
        
        This method:
        - Extracts model parameters from encoder output
        - Reconstructs metabolite signals
        - Adds macromolecular contributions
        - Adds B-spline baseline if enabled
        
        Args:
            enc: Encoder output
            cond: Conditional parameter for B-spline
            
        Returns:
            tuple: Reconstructed signal components
        """
        fr, damp,ph,ample_met,ample_MM,mm_f,mm_damp, mm_phase, spline_coeff = self.get_model_parameters(enc)
        # damp = torch.clamp(damp, max=30)
        dec = self.lc_met(fr, damp,ph,ample_met,ample_MM,mm_f,mm_damp, mm_phase)
        mm_rec = torch.Tensor(0)
        b_spline_rec = 0
        if self.param.parameters['spline']:
            b_spline_rec_im = 0
            if self.param.in_shape == 'real':
                b_spline_rec = self.bspline(spline_coeff[:, 0:self.param.parameters['numofsplines']], cond)
            if self.param.in_shape != 'real':
                b_spline_rec = self.bspline(spline_coeff[:, 0:self.param.parameters['numofsplines']//2], cond)
                b_spline_rec_im = self.bspline(spline_coeff[:,self.param.parameters['numofsplines']//2:self.param.parameters['numofsplines']], cond)
        if self.param.MM:
            mm_rec,mm_rec_param = self.lc_mm(fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase)

        return fr, damp, ph, mm_rec+mm_rec_param, dec, ample_met, mm_phase,b_spline_rec,spline_coeff,b_spline_rec_im

    def get_model_parameters(self,enc):
        """
        Extract model parameters from encoder output.
        
        Args:
            enc: Encoder output tensor
            
        Returns:
            tuple: Extracted parameters (fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase, spline_coeff)
        """
        fr = torch.unsqueeze(enc[:, -3],1)
        damp = torch.unsqueeze(enc[:, -2],1)
        ph = torch.unsqueeze(enc[:, - 1],1)
        ample_met =(enc[:, 0:(self.param.numOfSig)])

        ample_MM = (enc[:, (self.param.numOfSig):(self.param.numOfSig)+self.param.numOfMM])
        mm_f = enc[:, (self.param.numOfSig)+self.param.numOfMM:(self.param.numOfSig)+self.param.numOfMM*2]
        mm_phase = enc[:, (self.param.numOfSig)+self.param.numOfMM*2:(self.param.numOfSig)+self.param.numOfMM*3]
        mm_damp = enc[:, (self.param.numOfSig)+self.param.numOfMM*3:(self.param.numOfSig)+self.param.numOfMM*4]
            # torch.unsqueeze(enc[:, (self.param.numOfSig)+self.param.numOfMM*3],1)
        spline_coeff = 0
        if self.param.parameters['spline']:
            spline_coeff = enc[:,
                      (self.param.numOfSig) + self.param.numOfMM * 4:self.param.parameters['numofsplines']+(self.param.numOfSig) + self.param.numOfMM * 4]
        return fr, damp,ph,ample_met,ample_MM,mm_f,mm_damp, mm_phase,spline_coeff
    def lc(self,fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase):
        """
        Linear combination of metabolite and macromolecular signals.
        
        Args:
            fr: Frequency shift
            damp: Damping factor
            ph: Phase
            ample_met: Metabolite amplitudes
            ample_MM: Macromolecular amplitudes
            mm_f: Macromolecular frequencies
            mm_damp: Macromolecular damping
            mm_phase: Macromolecular phase
            
        Returns:
            torch.Tensor: Combined signal
        """
        # params = self.get_model_parameters(enc)
        if self.param.MM:
            a, b = self.lc_mm(fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase)
            out=(self.lc_met(fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase)
                 + a + b).real
        else:
            out=self.lc_met(fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase)
        return out.real
    def lc_met(self,fr, damp,ph,ample_met,ample_MM,mm_f,mm_damp, mm_phase):
        """
        Linear combination of metabolite signals.
        
        Args:
            fr: Frequency shift
            damp: Damping factor
            ph: Phase
            ample_met: Metabolite amplitudes
            ample_MM: Macromolecular amplitudes (unused in this method)
            mm_f: Macromolecular frequencies (unused in this method)
            mm_damp: Macromolecular damping (unused in this method)
            mm_phase: Macromolecular phase (unused in this method)
            
        Returns:
            torch.Tensor: Metabolite signal
        """
        sSignal = torch.matmul(ample_met[:, 0:(self.param.numOfSig)] + 0 * 1j, self.basis.T)
        dec = torch.multiply(sSignal, torch.exp(-2 * math.pi * (fr) * self.t.T * 1j))
        dec = torch.multiply(dec, torch.exp((-1 * damp) * self.t.T))
        dec = (dec * torch.exp(ph * 1j))
        return dec
    def lc_mm(self,f, damp,phase,ample_met, ample_MM,mm_f,mm_damp, mm_phase):
        mm_rec_param = 0
        mm_rec = 0
        if (self.param.MM == True):
            if self.param.MM_type == 'single' or self.param.MM_type == 'single_param':
                mm_enc = (ample_MM)
                mm_rec = (mm_enc[:]) * self.mm
                mm_rec = torch.multiply(mm_rec, torch.exp(-2 * math.pi * (mm_f) * self.t.T * 1j))
                mm_rec = torch.multiply(mm_rec, torch.exp((-1 * mm_damp) * self.t.T))
                mm_rec = mm_rec * torch.exp(mm_phase * 1j)
            if self.param.MM_type == 'param':
                # if self.param.MM_fd_constr == False:
                #     mm_enc = (ample_MM[:, 0:(self.param.numOfMM)])
                #     for idx in range(0, len(self.param.MM_f)):
                #         mm_rec_param += self.MM_model((mm_enc[:, idx].unsqueeze(1)), torch.unsqueeze(mm_f[:,idx],1),
                #                                 torch.unsqueeze(mm_damp[:,idx],1), torch.unsqueeze(mm_phase[:,idx],1), self.param.trnfreq * (self.param.MM_f[idx]),
                #                                 self.param.MM_a[idx], self.param.MM_d[idx])
                # else:
                mm_enc = (ample_MM[:, 0:(self.param.numOfMM)])
                for idx in range(0, len(self.param.MM_f)):
                    mm_rec_param += self.MM_model((mm_enc[:,idx].unsqueeze(1)), f,
                                                damp, torch.tensor(0), self.param.trnfreq * (self.param.MM_f[idx]),
                                                  self.param.MM_a[idx], self.param.MM_d[idx])
                if self.param.MM_conj:
                    mm_rec_param = torch.conj(mm_rec_param)

            if self.param.MM_type == 'combined':
                mm_enc = (ample_MM[:,0])
                mm_rec = (mm_enc.unsqueeze(1)) * self.mm
                mm_rec = torch.multiply(mm_rec, torch.exp(-2 * math.pi * (mm_f[:,0].unsqueeze(1)) * self.t.T * 1j))
                mm_rec = torch.multiply(mm_rec, torch.exp((-1 * mm_damp[:,0].unsqueeze(1)) * self.t.T))
                mm_rec = mm_rec * torch.exp(mm_phase[:,0].unsqueeze(1) * 1j)

                ample_MM=ample_MM[:,1:]
            # if self.param.MM_fd_constr == False:
            #     mm_enc = (ample_MM[:, 0:(self.param.numOfMM)])
            #     for idx in range(0, len(self.param.MM_f)):
            #         mm_rec_param += self.MM_model((mm_enc[:, idx].unsqueeze(1)), torch.unsqueeze(mm_f[:,idx],1),
            #                                 torch.unsqueeze(mm_damp[:,idx],1), torch.unsqueeze(mm_phase[:,idx],1), self.param.trnfreq * (self.param.MM_f[idx]),
            #                                 self.param.MM_a[idx], self.param.MM_d[idx])
            # else:
                mm_enc = (ample_MM[:, 0:(self.param.numOfMM)])
                for idx in range(0, len(self.param.MM_f)):
                    mm_rec_param += self.MM_model((mm_enc[:,idx].unsqueeze(1)), f,
                                                damp, torch.tensor(0), self.param.trnfreq * (self.param.MM_f[idx]),
                                                  self.param.MM_a[idx], self.param.MM_d[idx])
                if self.param.MM_conj:
                    mm_rec_param = torch.conj(mm_rec_param)

        return mm_rec, mm_rec_param


    def forward(self, x, cond = 0):
        decoded = self.param.inputSig(x) # B,C,Freq  B,2,Time
        # embed_cond = self.embed.weight[cond].unsqueeze(0)
        # cond_mean, cond_sd = torch.chunk(embed_cond, 2, 1)
        # decoded = decoded * cond_sd + cond_mean
        latent = self.met(decoded)
        if self.cond_max != 1:
            embeded = self.embed(torch.Tensor([cond]).to(self.embed.weight.device).unsqueeze(0))
            latent = latent @ embeded[:, :].view(1, latent.shape[2], latent.shape[
                2])  # + embeded[:,-7:].repeat(latent.shape[0],latent.shape[1],1)
            # latent = self.embed[cond](latent)
        # print(cond)
        # embed_cond = self.one_hot[cond]
        # enct = self.met.regres(torch.concatenate([latent.flatten(1),embed_cond.unsqueeze(0).repeat(x.shape[0],1)],1))

        enct = self.met.regres(latent)



        if self.beta != 0:
            soft_0 = self.act(enct[:, 0:self.param.numOfSig+self.param.numOfMM])
            soft_1 = (enct[:, self.enc_out:self.enc_out+self.param.numOfSig+self.param.numOfMM])
            enct = torch.concatenate(([soft_0,(enct[:, self.param.numOfSig+self.param.numOfMM:self.enc_out]),soft_1,(enct[:, self.enc_out+self.param.numOfSig+self.param.numOfMM:])]),1)
            enc = self.reparameterize(enct[:, 0:self.enc_out],enct[:, self.enc_out:2*(self.enc_out)])
        else:
            soft_0 = self.act(enct[:, 0:self.param.numOfSig+self.param.numOfMM])
            enct = torch.concatenate(([soft_0,(enct[:, self.param.numOfSig+self.param.numOfMM:self.enc_out])]),1)
            enc = enct
        # enc = torch.cat((enc,enct[:, 2*(self.param.numOfSig):]),dim=1)

        if self.param.MM_constr == True:
            self.tr = (self.sigm(enc[:,-1] - 5))
            fr, damp, ph, mm_rec, dec, ample_met, mm_phase,b_spline_rec,spline_coeff,b_spline_rec_im = self.model_decoder(enc[:, 0:-1],cond)
        else:
            fr, damp, ph, mm_rec, dec,ample_met, mm_phase,b_spline_rec,spline_coeff,b_spline_rec_im = self.model_decoder(enc,cond)

        if self.param.parameters["decode"]:
            decoded = self.decode(latent)

        if self.param.MM:
            dect = dec + mm_rec
        else:
            dect = dec
        return dect, enct, ample_met, fr, damp, ph, mm_rec, dec, decoded,b_spline_rec, spline_coeff,b_spline_rec_im
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        std = args[3]
        mm = args[4]
        decoded = args[5]
        ampl_p = args[6]
        ampl_l = args[7]
        b_spline_rec_real = args[8]
        spline_coeff = args[9]
        cond = args[10]
        damp = args[11]
        b_spline_rec_img = args[12]
        # Account for the minibatch samples from the dataset
        # cut_signal = args[7]
        # cut_dec =args[8]
        met_loss = 0
        loss_real = 0
        loss_imag = 0
        if (self.param.MM == True) and  (self.param.MM_constr == True):
            cut_signal = (((1 + self.sign(self.t - self.tr, 0.0000001)) / 2)[0:self.param.truncSigLen].T * input[:, 0:self.param.truncSigLen])
            cut_dec = (((1 + self.sign(self.t - self.tr, 0.0000001)) / 2).T * recons.clone())
            loss_real = self.criterion(cut_dec.real[:, 0:self.param.truncSigLen],
                                       cut_signal.real[:, 0:self.param.truncSigLen])
            loss_imag = 0
            if self.param.in_shape != 'real':
                loss_imag = self.criterion(cut_dec.imag[:, 0:self.param.truncSigLen],
                                       cut_signal.imag[:, 0:self.param.truncSigLen])
            met_loss = (loss_real + loss_imag) / (2 * self.param.truncSigLen)
            self.log("met_loss", met_loss.detach(),on_step=False,on_epoch=True)
            # reg = (tri / (self.param.truncSigLen))
            self.tr_ = torch.mean(self.tr)
            self.log("reg", self.tr_.detach(),on_step=False,on_epoch=True)
            met_loss= (met_loss + (self.tr_) * self.tr_wei * (self.param.batchsize))*0.5
            self.log("train_los", met_loss.detach(),on_step=False,on_epoch=True)

        if self.param.MM:
            recons += mm
        init_point = 1
        div_fac = 1
        if self.param.parameters['fbound'][0]:
            if self.param.parameters['zero_fill'][0] == True:
                recons = zero_fill_torch(recons,1,self.param.parameters['zero_fill'][1])
                input = zero_fill_torch(input, 1, self.param.parameters['zero_fill'][1])
            recons_f = torch.fft.fftshift(torch.fft.fft(recons[:,:self.param.truncSigLen], dim=1), dim=1)
            input_f = torch.fft.fftshift(torch.fft.fft(input, dim=1), dim=1)
            p1 = int(ppm2p(self.param.trnfreq,self.param.t_step,self.param.parameters['fbound'][2], (self.param.truncSigLen)))
            p2 = int(ppm2p(self.param.trnfreq,self.param.t_step,self.param.parameters['fbound'][1], (self.param.truncSigLen)))
            loss_real = self.criterion(recons_f.real[:, p1:p2]+b_spline_rec_real,
                                       input_f.real[:, p1:p2])
            loss_imag = 0
            if self.param.in_shape != 'real':
                loss_imag = self.criterion(recons_f.imag[:, p1:p2]+b_spline_rec_img,
                                           input_f.imag[:, p1:p2])
                div_fac  = 2
            recons_loss = (loss_real + loss_imag)/div_fac
        else:
            loss_real = self.criterion(recons.real[:, init_point:self.param.truncSigLen], input.real[:, init_point:self.param.truncSigLen])
            loss_imag = 0
            if self.param.in_shape != 'real':
                loss_imag = self.criterion(recons.imag[:, init_point:self.param.truncSigLen], input.imag[:, init_point:self.param.truncSigLen])
                div_fac = 2
            recons_loss = (loss_real+loss_imag)/div_fac
        self.log("recons_loss", recons_loss.detach(),on_step=False,on_epoch=True)
        if self.cond_max==1:
            cond=1
        #
        spline_loss = 0
        if self.param.parameters['spline']:
            spline_loss=(cond)* torch.linalg.vector_norm(spline_coeff[:,:-1]-spline_coeff[:,1:])#
            self.log("spline_loss", self.param.parameters['spline_reg'][self.ens_indx] * spline_loss.detach(),on_step=False,on_epoch=True)#self.param.parameters['spline_reg']*
        loss = met_loss + recons_loss + self.param.parameters['spline_reg'][self.ens_indx] * spline_loss#/(spline_coeff.shape[0]/32)
        #0.0001 * damp +
        if self.beta!=0:
            # - mu ** 2
            # kld_loss = torch.mean(-0.5 * torch.sum(-1+log_var-np.log(1e-5)+((1e-5)/log_var.exp()), dim=1), dim=0)
            # var = std**2
            # kld_loss = torch.mean(-0.5 * torch.sum(-1 + torch.log(var) - var, dim=1), dim=0)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + std - std.exp(), dim=1), dim=0)

            self.log("nll_los", kld_loss.detach(),on_step=False,on_epoch=True)
            # self.log("nll_los", kld_loss)
            beta_func = self.sigmoid(torch.tensor(self.global_step),1/2500,(1 * self.param.beta_step))
            # beta_func = torch.sigmoid((-10 + torch.tensor(self.global_step / (1 * self.param.beta_step)))*2)
            self.log("beta",beta_func.detach() * self.beta,on_step=False,on_epoch=True)
            beta_func=1
            loss =  loss + beta_func * self.beta * kld_loss
        if self.param.parameters["decode"] == True:
            if self.param.parameters['fbound'][0]:
                decoded_ = decoded[:, 0, 0:self.param.truncSigLen] + 1j*decoded[:, 1, 0:self.param.truncSigLen]
                recons_f = torch.fft.fftshift(torch.fft.fft(decoded_[:, :self.param.truncSigLen], dim=1), dim=1)
                input_f = torch.fft.fftshift(torch.fft.fft(input, dim=1), dim=1)
                p1 = int(self.param.ppm2p(self.param.parameters['fbound'][2], (self.param.truncSigLen)))
                p2 = int(self.param.ppm2p(self.param.parameters['fbound'][1], (self.param.truncSigLen)))
                loss_real = self.criterion(recons_f.real[:, p1:p2],
                                           input_f.real[:, p1:p2])
                loss_imag = 0
                if self.param.in_shape != 'real':
                    loss_imag = self.criterion(recons_f.imag[:, p1:p2],
                                               input_f.imag[:, p1:p2])
                    div_fac = 2
                recons_net_loss = (loss_real + loss_imag) / (div_fac * self.param.truncSigLen)
                self.log("recons_net_loss", recons_net_loss.detach(),on_step=False,on_epoch=True)
                loss += 1 * recons_net_loss
            else:
                loss_real_ = self.criterion(decoded[:, 0, init_point:self.param.truncSigLen],
                                            input.real[:, init_point:self.param.truncSigLen])
                loss_imag_ = 0
                if self.param.in_shape != 'real':
                    loss_imag_ = self.criterion(decoded[:, 1, init_point:self.param.truncSigLen],
                                                input.imag[:, init_point:self.param.truncSigLen])
                recons_net_loss = (loss_real_ + loss_imag_) / (div_fac  * self.param.truncSigLen)
                self.log("recons_net_loss", recons_net_loss.detach(),on_step=False,on_epoch=True)
                loss += 1*recons_net_loss
        if self.param.parameters["supervised"] is True:
            supervision_loss = self.r2(ampl_l,ampl_p)/self.param.numOfSig
            # supervision_loss = self.criterion(ampl_p, ampl_l)/(self.param.numOfSig*ampl_l.shape[0])
            self.log("supervision_loss", supervision_loss.detach(),on_step=False,on_epoch=True)
            loss += supervision_loss
        return loss,recons_loss
    def training_step(self, batch, batch_idx):
        if self.param.parameters["simulated"] is False:
            x = batch[0]
            ampl_batch = 0
        else:
            x, label = batch[0],batch[1]
            ampl_batch, alpha_batch = label[:, 0:-1], label[:, -1]

        x = self.getaug_vmap(x,
                              self.param.parameters['aug_params'][0],
                              self.param.parameters['aug_params'][1],
                              self.param.parameters['aug_params'][2],
                              self.param.parameters['aug_params'][3])
        cond = random.randint(0,self.cond_max-1)
        dec_real, enct, enc,_,damp,_,mm,dec,decoded,b_spline_rec,spline_coeff,b_spline_rec_im = self(x,cond)
        # self.param.parameters['spline_reg'] = cond
        # mu = enct[:, 0:self.param.numOfSig]
        # logvar = enct[:, self.param.numOfSig:2*(self.param.numOfSig)]
        mu = enct[:, 0:self.enc_out]
        logvar = enct[:, self.enc_out:2*(self.enc_out)]
        loss_mse,recons_loss = self.loss_function(dec, x, mu,logvar,mm,decoded,ampl_batch,enc,b_spline_rec,spline_coeff,cond,damp.mean(),b_spline_rec_im)
        self.training_step_outputs.append(loss_mse.detach())
        self.log('damp',damp.mean().detach(),on_step=False,on_epoch=True)
        return {'loss': loss_mse,'recons_loss':recons_loss}

    def validation_step(self, batch, batch_idx):
        # self.do_valid()

        r2 = 0
        corr = 0
        if self.param.parameters["simulated"] is False:
            x = batch[0]
        else:
            x, label = batch[0], batch[1]
            ampl_batch, alpha_batch = label[:,0:-1],label[:,-1]
            _, _, enc, _, alpha, _, mm, dec, decoded= self(x)
            r2 = self.r2score(ampl_batch[:, 0:self.param.numOfSig], enc[:, 0:self.param.numOfSig])
            corr = self.pearsoncorr(ampl_batch[:, 0:self.param.numOfSig], enc[:, 0:self.param.numOfSig])
            # error = (ampl_batch[:,0:self.param.numOfSig] - enc[:,0:self.param.numOfSig])
            # mean = torch.mean(ampl_batch[:,0:self.param.numOfSig],dim=0)
            # stot = (ampl_batch[:,0:self.param.numOfSig] - mean)
            # r2 = 1-(torch.sum(error**2,dim=0)/torch.sum(stot**2,dim=0))
        with torch.no_grad():
            results = self.training_step(batch, batch_idx)
        try:
            if (self.current_epoch % self.param.parameters['val_freq'] == 0 and batch_idx == 0):
                # id = int(np.random.rand() * 300)
                id = 120
                # sns.scatterplot(x=alpha_batch.cpu(), y=error.cpu())
                # sns.scatterplot(x=10*ampl_batch[:,12].cpu(),y=10*enc[:,12].cpu())
                # # plt.title(str(r2))
                # plt.show()
                # ampl_t = min_c + np.multiply(np.random.random(size=(1, 21)), (max_c - max_c))
                # y_n, y_wn = getSignal(ampl_t, 0, 5, 0, 0.5)
                rang = [1, 5]
                # id= 10
                fig = plt.figure()
                # plotppm(np.fft.fftshift(np.fft.fft((y_n.T)).T), 0, 5,False, linewidth=0.3, linestyle='-')
                p1 = int(ppm2p(self.param.trnfreq,self.param.t_step,rang[0], (self.param.y_test_trun.shape[1])))
                p2 = int(ppm2p(self.param.trnfreq,self.param.t_step,rang[1], (self.param.y_test_trun.shape[1])))
                plotppm(np.fft.fftshift(np.fft.fft((self.param.y_test_trun[id, :])).T)[p2:p1], rang[0], rang[1], False, linewidth=0.3, linestyle='-',label="y_test")
                cond_ = 1#random.randint(0,self.cond_max-1)
                # self.param.parameters['spline_reg'] = cond_
                # plt.plot(np.fft.fftshift(np.fft.fft(np.conj(y_trun[id, :])).T)[250:450], linewidth=0.3)
                with torch.no_grad():
                    rec_signal,_,enc, fr, damp, ph,mm_v,_,decoded, spline_rec,spline_coeff,_ = self(torch.unsqueeze(self.param.y_test_trun[id, :], 0).to(self.param.parameters['gpu']),cond_)
                # spline_loss = torch.linalg.vector_norm(spline_coeff[:, :-1] - spline_coeff[:, 1:])  #
                # self.log(f"spline_loss_val_{cond_}", spline_loss)
                # plotppm(np.fft.fftshift(np.fft.fft(((rec_signal).cpu().detach().numpy()[0,0:truncSigLen])).T), 0, 5,False, linewidth=1, linestyle='--')
                if self.param.parameters["decode"] == True:
                    if self.param.in_shape == 'real':
                        decoded = decoded[:,0,:]
                        plotppm(40+np.fft.fftshift(np.fft.rfft(
                            (decoded.cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T)[p2:p1], rang[0], rang[1],
                                True, linewidth=1, linestyle='--',label="decoded")
                    else:
                        decoded = decoded[:,0,:] + decoded[:,1,:]*1j
                        plotppm(40+np.fft.fftshift(np.fft.fft(
                            (decoded.cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T)[p2:p1], rang[0], rang[1],
                                True, linewidth=1, linestyle='--',label="decoded")
                rec_sig = rec_signal.cpu().detach().numpy()[0, 0:self.param.truncSigLen]


                plotppm(np.fft.fftshift(np.fft.fft(
                    (rec_sig)).T)[p2:p1], rang[0], rang[1],
                        False, linewidth=1, linestyle='--',label="rec_sig")
                plt.title("#Epoch: " + str(self.current_epoch))
                plt.legend()
                # savefig(self.param,self.param.epoch_dir+"decoded_paper1_1_epoch_" + "_"+ str(self.tr_wei))
                self.logger.log_image(key="Fit", images=[fig])
                # self.param.savefig(
                #     self.param.epoch_dir + "decoded_paper1_1_epoch_" + str(self.current_epoch) + "_" + str(self.tr_wei))
                plt.close()

                fig = plt.figure()
                if self.param.MM == True:
                    plotppm(5+10*np.fft.fftshift(np.fft.fft(((mm_v).cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T)[p2:p1], rang[0], rang[1], False, linewidth=1,linestyle='--',label="mm_v")

                # self.param.plotppm(30+np.fft.fftshift(np.fft.fft(
                #     (rec_sig)).T)[p2:p1], rang[0], rang[1],
                #         False, linewidth=1, linestyle='--',label="rec_sig")

                # self.param.plotppm(np.fft.fftshift(np.fft.fft((rec_sig[0:self.param.y_test_trun.shape[1]]-self.param.y_test_trun.numpy()[id, :])).T)[p2:p1]
                #                    , rang[0], rang[1], True,
                #                    linewidth=0.3, linestyle='-',label="rec_sig-y_test_trun")


                # self.param.plotppm(200 + np.fft.fftshift(np.fft.fft(
                #     (self.param.y_test_trun[id, :]-rec_signal.cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T), rang[0], rang[1],
                #         True, linewidth=1, linestyle='--')
                sns.despine()
                plot_basis(self.param.trnfreq,self.param.t_step, self.param.basisset, self.param.t, self.param.met_name, 10*(enc).cpu().detach().numpy(),
                           fr.cpu().detach().numpy(), damp.cpu().detach().numpy(),
                           ph.cpu().detach().numpy(),rng=rang)
                # plt.plot(np.fft.fftshift(np.fft.fft(np.conj(rec_signal.cpu().detach().numpy()[0,0:trunc])).T)[250:450], linewidth=1,linestyle='--')
                plt.title("#Epoch: " + str(self.current_epoch))
                plt.legend()
                # self.param.savefig(self.param.epoch_dir+"fit_paper1_1_epoch_" + str(self.current_epoch) +"_"+ str(self.tr_wei))
                # savefig(self.param,
                #     self.param.epoch_dir + "fit_paper1_1_epoch_" + "_" + str(self.tr_wei))
                plt.tight_layout()
                self.logger.log_image(key="Basis", images=[fig])
                plt.close()
                rang = [1.8, 3.8]
                p1 = int(ppm2p(self.param.trnfreq,self.param.t_step,rang[0], (self.param.y_test_trun.shape[1])))
                p2 = int(ppm2p(self.param.trnfreq,self.param.t_step,rang[1], (self.param.y_test_trun.shape[1])))
                if self.param.parameters['spline']:
                    # rec_signal, _, enc, fr, damp, ph, mm_v, _, decoded, spline_rec, spline_coeff = self(
                    #     torch.unsqueeze(self.param.y_test_trun[id, :], 0).cuda(), cond_)
                    fig = plt.figure()
                    spline_rec = spline_rec.cpu().detach()
                    plt.plot(spline_rec.T)
                    # self.param.savefig(self.param.epoch_dir+"spline")
                    y_test = (self.param.y_test_trun[id, :]).unsqueeze(0)
                    y_test = zero_fill_torch(y_test, 1, self.param.parameters['zero_fill'][1]).cpu().detach()
                    plt.plot(torch.fft.fftshift(torch.fft.fft((y_test)).T)[self.p1:self.p2])
                    rec_sig = zero_fill_torch(rec_signal, 1, self.param.parameters['zero_fill'][1]).cpu().detach()
                    plt.plot(torch.fft.fftshift(torch.fft.fft((rec_sig)).T)[self.p1:self.p2] + spline_rec.cpu().detach().T)
                    mm_v = zero_fill_torch(mm_v, 1, self.param.parameters['zero_fill'][1]).cpu().detach()
                    plt.plot(torch.fft.fftshift(torch.fft.fft((mm_v)).T)[self.p1:self.p2])
                    plt.title(f'condition:{cond_}')
                    # savefig(self.param, self.param.epoch_dir + "result")
                    # tensorboard = self.logger.experiment
                    # tensorboard.add_figure("recons", fig, self.current_epoch)
                    self.logger.log_image(key="samples", images=[fig])
                    plt.close()
        except:
            print("problem in plotting during validation")
        self.log("val_acc", results['loss'].detach(),on_step=False,on_epoch=True)
        self.log("val_recons_loss", results['recons_loss'].detach(),on_step=False,on_epoch=True)
        self.validation_step_outputs.append(results['loss'].detach())
        return r2,corr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.param.parameters['lr'])
        # if self.param.parameters['reduce_lr'][0] ==True:
        lr_scheduler = {
            # 'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=self.param.parameters['reduce_lr'][0]),
            # 'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.param.max_epoch),
            # 'scheduler': torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99),
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=[int(self.param.max_epoch/8),int(self.param.max_epoch*0.9)],
                                                              gamma=self.param.parameters['reduce_lr'][0]),
            # 'monitor':self.param.parameters['reduce_lr'][1],
            'name': 'scheduler'
        }
        return [optimizer],[lr_scheduler]

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log("epoch_los",avg_loss.detach(),on_step=False,on_epoch=True)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        r2 = []
        corr = []
        if self.param.parameters["simulated"] is True:
            for list in self.validation_step_outputs:
                r2.append((list[0]))
                corr.append((list[1]))
            try:
                r2 = torch.mean(torch.stack(r2),axis=0)
                corr = torch.mean(torch.stack(corr), axis=0)
                performance = 0
                for idx,name in enumerate(self.param.met_name):
                    self.log(name,r2[idx],on_step=False,on_epoch=True)
                for name in self.selected_met:
                    performance+=r2[self.param.met_name.index(name)]
                performance=performance/len(self.selected_met)
                self.log("performance",performance,on_step=False,on_epoch=True)
                r2_total = torch.mean(r2)
                self.log("r2_total",r2_total,on_step=False,on_epoch=True)
                corr_total = torch.mean(corr)
                self.log("corr_total",corr_total,on_step=False,on_epoch=True)
            except:
                pass
        self.validation_step_outputs.clear()

    def r2(self,output, target):
        target_mean = torch.mean(target,0)
        ss_tot = torch.sum((target - target_mean) ** 2,0)
        ss_res = torch.sum((target - output) ** 2,0)
        r2 = (ss_res+1e-10) / (ss_tot+1e-10)
        return torch.mean(r2[r2<1])

    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.met.encoder.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def crlb(self, x, noise_sd, ampl=None, percent = True, cal_met=True):
        self.eval()
        if cal_met == True:
            x = torch.unsqueeze(x, 0)
            enc, latent = self.met(self.param.inputSig(x))
        else:
            enc = ampl

        if len(enc.shape)==1:
            enc = enc.unsqueeze(0)
        fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase, spline_coeff = self.get_model_parameters(enc)
        D= jacfwd(self.lc,argnums=(3,4))(fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase)
        # D_ = torch.stack(D[1:])
        D_ = torch.transpose(torch.squeeze(torch.concatenate(D,-1)),1,0)
        I = 1 / noise_sd ** 2 * torch.einsum('mk,nk', D_, D_)
        I_inv = torch.inverse(I)
        crlb = torch.sqrt(I_inv.diag())
        if percent == True:
            crlb = crlb/torch.abs(enc[:,0:-1])
        return crlb

    def bspline(self,coeff,cond):
        # cond = self.param.parameters['spline_reg']
        if self.cond_max != 1:
            coeff = self.bspline_embed[cond](coeff)
        length, batch = coeff.permute(1,0).shape
        # length = int(length/(cond+1))
        # coeff_ = coeff[:,0::(cond+1)]
        dx = ((self.in_size)/(length-2))
        t = torch.linspace(0, (self.in_size) + dx, length).to(coeff.device)
        # t = t[0::(cond+1)]
        coeffs = natural_cubic_spline_coeffs(t, coeff.permute(1,0))
        spline = NaturalCubicSpline(coeffs)
        # offset = ((self.in_size) + 2*((self.in_size)/(length-2)))/(length-2)
        t = torch.linspace(dx, dx+self.in_size-1, self.in_size).to(coeff.device)
        out = spline.evaluate(t)
        return out.T

    def get_augment(self, signal, f_band, ph_band, d_band, noise_level):

        shift = f_band * torch.rand(1) - (f_band / 2)
        ph = ph_band * torch.rand(1) * math.pi - ((ph_band / 2) * math.pi)
        d = torch.rand(1) * d_band

        freq = -2 * math.pi * shift

        y = signal * torch.exp(1j * (ph.to(signal.device) + freq.to(signal.device) * self.t.T))  # t is the time vector
        y = y * torch.exp(-d.to(signal.device) * self.t.T)

        noise = (torch.randn(1, len(signal)) +
                 1j * torch.randn(1, len(signal))) * noise_level  # example noise level
        return (y + noise.to(signal.device) ).squeeze()

    def do_valid(self):


        affine_id = "C:\\Work\\nmrlab10\\MRSI_fit\\data\\new_data\\csi_template_mod.nii"
        test_id = "C:\\Work\\nmrlab10\\MRSI_fit\\data\\new_data\\HC08_M01.npy"
        mask_id = "C:\\Work\\nmrlab10\\MRSI_fit\\data\\new_data\\HC08_M01_mask.npy"
        subj = ["HC08_M01"][0]
        names = [["Gln"]]
        if affine_id is not None:
            nifti_templ = nib.load(affine_id)
            affine = nifti_templ.affine
        else:
            affine = np.eye(4)
        # self.parameters['test_subjs']:
        temp = 'version_p'
        # path = f'test\\{os.path.basename(test_id[:-4])}\\{self.parameters["cond_test"]}\\'
        # Path(self.param.saving_dir + path).mkdir(exist_ok=True, parents=True)
        data = np.load(test_id)  # f"data/8 subj/MS_Patients_3Dconcept/test_data_p{test_id}.npy")
        mask = np.load(mask_id)  # f'{test_id})#f"data/8 subj/MS_Patients_3Dconcept/test_mask_p{test_id}.npy")
        # data= np.load(f"data/8 subj/MRSI_8volunteers_raw_data/data_HC_from_eva/test_data_{test_id}.npy")
        # mask = np.load(f"data/8 subj/MRSI_8volunteers_raw_data/data_HC_from_eva/test_mask_{test_id}.npy")
        mask = mask.squeeze()
        y_n = data.squeeze()
        nx, ny, nz, nt = y_n.shape
        y_n = y_n[:,:,16][mask[:,:,16].astype(bool), :]

        y_test_np = y_n.astype('complex64')

        if self.param.data_conj == True:
            y_test_np = np.conj(y_test_np)
        else:
            y_test_np = y_test_np
        # y = y[4:,:]
        y_test_np,_ = normalize(y_test_np.T)
        ang = np.angle(y_test_np[1, :])
        y_test_np = y_test_np * np.exp(-1 * (ang+np.pi)  * 1j)

        y_test = torch.from_numpy(y_test_np.T[:, 0:self.param.org_truncSigLen]).to(self.param.parameters['gpu'])
        cond_ = 1#random.randint(0,self.cond_max-1)
        dect, enct, enc, fr, damp, ph, mm_rec, dec, decoded, spline, loss, _ = self.forward(y_test, cond=cond_)

        ref_dl = enc[:,self.param.met_name.index('Cr')] + enc[:,self.param.met_name.index('PCr')]

        for gp_name in names:
            dl = 0
            for name in gp_name:
                data_1 = enc[:, self.param.met_name.index(name)]
                data_1 = (data_1)
                dl += torch.nan_to_num(data_1)
        dl = torch.nan_to_num(dl / ref_dl)
        # lc_model_path = f'C:\\Work\\nmrlab10\\MRSI_fit\\data\\new_data\\{subj}\\maps\\Orig\\GPC+PCh_amp_map.nii'
        lc_model_path = f'C:\\Work\\nmrlab10\\MRSI_fit\\data\\new_data\\{subj}\\maps\\Ratio\\Gln_RatToCr+PCr_map.nii'
        lc_model_ = load_nifti_file(lc_model_path)
        lc_model_ = np.nan_to_num(lc_model_.get_fdata())
        lc_model_ = np.flip(lc_model_, axis=(0, 1))

        trsh = 2
        lc_model_[lc_model_ > trsh] = 0.0
        dl[dl > trsh] = 0.0

        tensor1_cpu = lc_model_[:,:,16]
        tensor2_cpu = np.zeros_like(mask[:,:,16])
        tensor2_cpu[mask[:,:,16].astype(bool)] = dl.cpu()

        # Compute the absolute difference between the tensors
        diff_tensor = np.abs(lc_model_[:,:,16] - tensor2_cpu)

        # Create a figure with subplots
        try:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))

            # Plot the first image
            im1 = axs[0].imshow(tensor1_cpu, cmap='gray')
            axs[0].set_title('LC')
            axs[0].axis('off')

            # Plot the second image
            axs[1].imshow(tensor2_cpu, cmap='gray')
            axs[1].set_title('DL')
            axs[1].axis('off')

            # Plot the difference image
            axs[2].imshow(diff_tensor, cmap='gray')
            axs[2].set_title('Difference')
            axs[2].axis('off')
            vmin = min(tensor1_cpu.min(), tensor2_cpu.min(), diff_tensor.min())
            vmax = max(tensor1_cpu.max(), tensor2_cpu.max(), diff_tensor.max())

            cbar_ax = fig.add_axes([0.9, 0.15, 0.01, 0.7])

            # Normalize the color bar
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            # Add color bar for all images
            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="gray"), cax=cbar_ax)
            # Adjust the spacing between subplots
            plt.tight_layout(rect=[0, 0, 0.9, 1])
            self.logger.log_image(key="compare", images=[fig])
            plt.close()
        except:
            pass
        # nifti_file_1_path_ref_1 = f'C:\\Work\\nmrlab10\\MRSI_fit\\data\\new_data\\{subj}\\maps\\Orig\\Cr_amp_map.nii'
        # nifti_file_1_path_ref_2 = f'C:\\Work\\nmrlab10\\MRSI_fit\\data\\new_data\\{subj}\\maps\\Orig\\PCr_amp_map.nii'
        # nifti_data_1_ref = np.nan_to_num(load_nifti_file(nifti_file_1_path_ref_1).get_fdata()) + np.nan_to_num(
        #     load_nifti_file(nifti_file_1_path_ref_2).get_fdata())
        #
        # for gp_name in names:
        #     nifti_data_1_ = 0
        #     nifti_data_2_ = 0
        #     for name in gp_name:
        #         nifti_file_1_path = f'C:\\Work\\nmrlab10\\MRSI_fit\\data\\new_data\\{subj}\\maps\\Orig\\{name}_amp_map.nii'
        #         nifti_img_1 = load_nifti_file(nifti_file_1_path)
        #         nifti_data_1 = np.nan_to_num(nifti_img_1.get_fdata())
        #         nifti_data_1 = np.nan_to_num(nifti_data_1 / nifti_data_1_ref)
        #         nifti_data_1_ += np.nan_to_num(nifti_data_1)
        #
        # nifti_data_1_ = np.flip(nifti_data_1_, axis=(0,1))
        nifti_data_1_ = torch.from_numpy(lc_model_[:,:,16][mask[:,:,16].astype(bool)]).to(self.device)

        # nifti_data_1_ = torch.clamp(nifti_data_1_, max=3.0)
        # dl = torch.clamp(dl, max=3.0)
        # trsh = 4




        mask_vol = ~((dl == 0.0) | (nifti_data_1_ == 0.0))
        try:
            r2, _ = pearsonr(nifti_data_1_[mask_vol].reshape(-1).cpu().detach().numpy(), dl[mask_vol].reshape(-1).cpu().detach().numpy())
        # r2 = calculate_r2_torch(nifti_data_1_[mask_vol],dl[mask_vol])
            self.log(f"r2_{names}",r2, on_step=False,on_epoch=True)
        except:
            print("error in calculating r2")
        x = nifti_data_1_[mask_vol].cpu().detach().numpy()
        y = dl[mask_vol].cpu().detach().numpy()

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create the scatter plot with custom aesthetics
        ax.scatter(x, y, color='#1f77b4', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)

        # Set labels and title
        ax.set_xlabel('Lcmodel Data', fontsize=14)
        ax.set_ylabel('DL Data', fontsize=14)
        ax.set_title('Scatter Plot of NIFTI Data vs DL Data', fontsize=16)

        # Add a legend
        ax.legend(['Data Points'], loc='upper right', fontsize=12)

        # Customize the plot appearance
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)

        # Adjust the padding between the plot and the edges of the figure
        plt.tight_layout()
        plt.title(cond_)
        self.logger.log_image(key="lc_model", images=[fig])
        plt.close()
        # print(dect)

def calculate_r2_torch(y_true, y_pred):
    """
    Calculates the R-squared (R^2) value for a set of predicted and actual values using PyTorch.

    Args:
        y_true (torch.Tensor): The actual values.
        y_pred (torch.Tensor): The predicted values.

    Returns:
        float: The R-squared value.
    """
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)

    r2 = 1 - (ss_residual / ss_total)

    return r2.item()
