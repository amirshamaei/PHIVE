import random

import torch
import torch.nn as nn
import pytorch_lightning as pl
import math
import numpy as np
import scipy.io as sio
import torchmetrics
from torch.func import jacfwd
from torchmetrics import R2Score, PearsonCorrCoef
from torch import Tensor
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from Models.UNET import ConvNet_ENC, ConvNet_DEC


from torch.autograd.functional import jacobian
from Models.transformer import Transformer, TransformerB
from torchcubicspline import(natural_cubic_spline_coeffs,
                             NaturalCubicSpline)

from utils.utils import ppm2p, zero_fill_torch


class Encoder_Model(pl.LightningModule):
    def __init__(self,depth, beta, tr_wei, i,param):
        super().__init__()
        self.ens_indx = i
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.param = param
        self.met = []
        self.selected_met = ["Cr", "GPC", "sIns", "NAA", "PCho", "Tau"]
        self.t = torch.from_numpy(param.t).float().cuda(self.param.parameters['gpu'])[0:self.param.org_truncSigLen]
        self.sw = 1/param.t_step
        self.basis = torch.from_numpy(param.basisset[:self.param.org_truncSigLen, 0:param.numOfSig].astype('complex64')).cuda(self.param.parameters['gpu'])
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False).cuda(self.param.parameters['gpu'])
        # self.r2 = torchmetrics.R2Score(adjusted=True)
        if self.param.MM_constr == True :
            print('tr is not in the model')
        else:
            self.tr = nn.Parameter(torch.tensor(0.004).cuda(self.param.parameters['gpu']), requires_grad=True)
        self.tr_wei = tr_wei
        self.act = nn.Softplus()
        self.lact = nn.ReLU6()
        self.sigm = nn.Sigmoid()
        self.model = None
        self.tanh = nn.Tanh()
        # self.mult_factor = 2
        if self.param.MM == True :
            if self.param.MM_type == 'single' or self.param.MM_type == 'single_param':
                self.enc_out =  1 * (param.numOfSig) + 4 +3 # numof metabolite + parameters + MM parameters
                self.mm = torch.from_numpy(param.mm[:self.param.org_truncSigLen,:].astype('complex64')).cuda(self.param.parameters['gpu']).T
            if self.param.MM_type == 'param':
                if self.param.MM_fd_constr == False:
                    self.enc_out =  (1* (self.param.numOfSig)+self.param.numOfMM*3 + 3 + (self.param.numOfMM))
                else:
                    self.enc_out =  1* (self.param.numOfSig)+self.param.numOfMM + 3 + 1 + 1 + 1
        else:
            self.enc_out = 1 * (param.numOfSig) + 3

        if self.param.MM_constr == True:
            self.enc_out += 1


        if self.param.parameters['spline']:
            self.enc_out = self.enc_out + self.param.parameters['numofsplines']


        try:
            self.dropout = param.parameters['dropout']
        except:
            self.dropout = 0

        if self.param.in_shape == 'real':
            self.in_chanel = 1
        else:
            self.in_chanel = 2


        if self.beta != 0:
            self.enc_out_ = 2 * self.enc_out
        else:
            self.enc_out_ = self.enc_out

        # self.embed = nn.Embedding(2, 128,_freeze=False)
        # self.cond_max = 2
        # self.embed = []
        # for i in range(self.cond_max):
        #     self.embed.append(nn.Linear(self.param.parameters['numofsplines'],self.param.parameters['numofsplines']).cuda(self.param.parameters['gpu']))
        # self.one_hot = torch.eye(self.cond_max,requires_grad=False).cuda(self.param.parameters['gpu'])
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
        if param.parameters['MM_model'] == "lorntz":
            self.MM_model = self.Lornz
        if param.parameters['MM_model'] == "gauss":
            self.MM_model = self.Gauss


            # self.decode = LinDec(depth, param.banorm, self.enc_out, self.param.truncSigLen)
        self.r2score = R2Score(num_outputs=self.param.numOfSig, multioutput='raw_values')
        self.pearsoncorr = PearsonCorrCoef(num_outputs=self.param.numOfSig)
        if self.param.parameters['zero_fill'][0] == True:
            self.param.truncSigLen = self.param.parameters['zero_fill'][1]
        if self.param.parameters['domain'] == 'freq':
            self.p1 = int(ppm2p(self.param,self.param.parameters['fbound'][2], (self.param.truncSigLen)))
            self.p2 = int(ppm2p(self.param,self.param.parameters['fbound'][1], (self.param.truncSigLen)))
            self.in_size = int(self.p2-self.p1)



    def sign(self,t,eps):
        return (t/torch.sqrt(t**2+ eps))

    def sigmoid(self,x,a, b):
        return (1/(1+torch.exp(-1*a*(x-b))))
    def Gauss(self, ampl, f, d, ph, Crfr, Cra, Crd):
        return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                       torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                  torch.exp(-(d+Crd)**2 * self.t.T*self.t.T))
    def Lornz(self, ampl, f, d, ph, Crfr, Cra, Crd):
        return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                       torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                  torch.exp(-(d+Crd) * self.t.T))
    def Voigt(self, ampl, f, dl,dg, ph, Crfr, Cra, Crd):
        return (Cra*ampl) * torch.multiply(torch.multiply(torch.exp(ph * 1j),
                       torch.exp(-2 * math.pi * ((f + Crfr)) * self.t.T * 1j)),
                                  torch.exp(-(((dl) * self.t.T)+(dg+Crd) * self.t.T*self.t.T)))
    def model_decoder(self,enc):
        fr, damp,ph,ample_met,ample_MM,mm_f,mm_damp, mm_phase, spline_coeff = self.get_model_parameters(enc)
        # damp = torch.clamp(damp, max=30)
        dec = self.lc_met(fr, damp,ph,ample_met,ample_MM,mm_f,mm_damp, mm_phase)
        mm_rec = torch.Tensor(0)
        b_spline_rec = 0
        if self.param.parameters['spline']:
            b_spline_rec = self.bspline(spline_coeff)
        if self.param.MM:
            mm_rec = self.lc_mm(fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase)

        return fr, damp, ph, mm_rec, dec, ample_met, mm_phase,b_spline_rec,spline_coeff

    def get_model_parameters(self,enc):
        fr = torch.unsqueeze(enc[:, -3],1)
        damp = torch.unsqueeze(enc[:, -2],1)
        ph = torch.unsqueeze(enc[:, -1],1)
        ample_met = torch.nn.functional.relu(enc[:, 0:(self.param.numOfSig)])

        ample_MM = torch.nn.functional.relu(enc[:, (self.param.numOfSig):(self.param.numOfSig)+self.param.numOfMM])
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
        # params = self.get_model_parameters(enc)
        if self.param.MM:
            out=(self.lc_met(fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase)
                 + self.lc_mm(fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase)).real
        else:
            out=self.lc_met(fr, damp, ph, ample_met, ample_MM, mm_f, mm_damp, mm_phase)
        return out.real
    def lc_met(self,fr, damp,ph,ample_met,ample_MM,mm_f,mm_damp, mm_phase):
        sSignal = torch.matmul(ample_met[:, 0:(self.param.numOfSig)] + 0 * 1j, self.basis.T)
        dec = torch.multiply(sSignal, torch.exp(-2 * math.pi * (fr) * self.t.T * 1j))
        dec = torch.multiply(dec, torch.exp((-1 * damp) * self.t.T))
        dec = (dec * torch.exp(ph * 1j))
        return dec
    def lc_mm(self,fr, damp,ph,ample_met, ample_MM,mm_f,mm_damp, mm_phase):
        if (self.param.MM == True):
            if self.param.MM_type == 'single' or self.param.MM_type == 'single_param':
                mm_enc = (ample_MM)
                mm_rec = (mm_enc[:]) * self.mm
                mm_rec = torch.multiply(mm_rec, torch.exp(-2 * math.pi * (mm_f) * self.t.T * 1j))
                mm_rec = torch.multiply(mm_rec, torch.exp((-1 * mm_damp) * self.t.T))
                mm_rec = mm_rec * torch.exp(mm_phase * 1j)
            if self.param.MM_type == 'param':
                mm_rec = 0
                if self.param.MM_fd_constr == False:
                    mm_enc = (ample_MM[:, 0:(self.param.numOfMM)])
                    for idx in range(0, len(self.param.MM_f)):
                        mm_rec += self.MM_model((mm_enc[:, idx].unsqueeze(1)), torch.unsqueeze(mm_f[:,idx],1),
                                                torch.unsqueeze(mm_damp[:,idx],1), torch.unsqueeze(mm_phase[:,idx],1), self.param.trnfreq * (self.param.MM_f[idx]),
                                                self.param.MM_a[idx], self.param.MM_d[idx])
                else:
                    mm_enc = (ample_MM[:, 0:(self.param.numOfMM)])
                    for idx in range(0, len(self.param.MM_f)):
                        mm_rec += self.MM_model((mm_enc[:,idx].unsqueeze(1)), fr,
                                                    damp, torch.tensor(0), self.param.trnfreq * (self.param.MM_f[idx]),
                                                      self.param.MM_a[idx], self.param.MM_d[idx])
                if self.param.MM_conj:
                    mm_rec = torch.conj(mm_rec)

        return mm_rec
    def forward(self, x, cond = 0):
        decoded = self.param.inputSig(x)

        latent = self.met(decoded)

        # embed_cond = self.embed.weight[cond]
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
            fr, damp, ph, mm_rec, dec, ample_met, mm_phase,b_spline_rec,spline_coeff = self.model_decoder(enc[:, 0:-1])
        else:
            fr, damp, ph, mm_rec, dec,ample_met, mm_phase,b_spline_rec,spline_coeff = self.model_decoder(enc)

        if self.param.parameters["decode"]:
            decoded = self.decode(latent)

        if self.param.MM:
            dect = dec + mm_rec
        else:
            dect = dec
        return dect, enct, ample_met, fr, damp, ph, mm_rec, dec, decoded,b_spline_rec,spline_coeff
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
        b_spline_rec = args[8]
        spline_coeff = args[9]
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
            self.log("met_loss", met_loss)
            # reg = (tri / (self.param.truncSigLen))
            self.tr_ = torch.mean(self.tr)
            self.log("reg", self.tr_)
            met_loss= (met_loss + (self.tr_) * self.tr_wei * (self.param.batchsize))*0.5
            self.log("train_los", met_loss)

        if self.param.MM:
            recons += mm
        init_point = 1
        div_fac = 1
        if self.param.parameters['fbound'][0]:
            if self.param.parameters['zero_fill'][0] == True:
                recons = zero_fill_torch(self,recons,1,self.param.parameters['zero_fill'][1])
                input = zero_fill_torch(self,input, 1, self.param.parameters['zero_fill'][1])
            recons_f = torch.fft.fftshift(torch.fft.fft(recons[:,:self.param.truncSigLen], dim=1), dim=1)
            input_f = torch.fft.fftshift(torch.fft.fft(input, dim=1), dim=1)
            p1 = int(ppm2p(self.param,self.param.parameters['fbound'][2], (self.param.truncSigLen)))
            p2 = int(ppm2p(self.param,self.param.parameters['fbound'][1], (self.param.truncSigLen)))
            loss_real = self.criterion(recons_f.real[:, p1:p2]+b_spline_rec,
                                       input_f.real[:, p1:p2])
            loss_imag = 0
            if self.param.in_shape != 'real':
                loss_imag = self.criterion(recons_f.imag[:, p1:p2],
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
        self.log("recons_loss", recons_loss)
        spline_loss=torch.linalg.vector_norm(spline_coeff[:,:-1]-spline_coeff[:,1:])#
        self.log("spline_loss", spline_loss)#self.param.parameters['spline_reg']*
        loss = met_loss + recons_loss + self.param.parameters['spline_reg'][self.ens_indx]*spline_loss#/(spline_coeff.shape[0]/32)

        if self.beta!=0:
            # - mu ** 2
            # kld_loss = torch.mean(-0.5 * torch.sum(-1+log_var-np.log(1e-5)+((1e-5)/log_var.exp()), dim=1), dim=0)
            # var = std**2
            # kld_loss = torch.mean(-0.5 * torch.sum(-1 + torch.log(var) - var, dim=1), dim=0)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + std - std.exp(), dim=1), dim=0)
            self.log("nll_los", kld_loss)
            # self.log("nll_los", kld_loss)
            beta_func = self.sigmoid(torch.tensor(self.global_step),1/2500,(1 * self.param.beta_step))
            # beta_func = torch.sigmoid((-10 + torch.tensor(self.global_step / (1 * self.param.beta_step)))*2)
            self.log("beta",beta_func * self.beta)
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
                self.log("recons_net_loss", recons_net_loss)
                loss += 1 * recons_net_loss
            else:
                loss_real_ = self.criterion(decoded[:, 0, init_point:self.param.truncSigLen],
                                            input.real[:, init_point:self.param.truncSigLen])
                loss_imag_ = 0
                if self.param.in_shape != 'real':
                    loss_imag_ = self.criterion(decoded[:, 1, init_point:self.param.truncSigLen],
                                                input.imag[:, init_point:self.param.truncSigLen])
                recons_net_loss = (loss_real_ + loss_imag_) / (div_fac  * self.param.truncSigLen)
                self.log("recons_net_loss", recons_net_loss)
                loss += 1*recons_net_loss
        if self.param.parameters["supervised"] is True:
            supervision_loss = self.r2(ampl_l,ampl_p)/self.param.numOfSig
            # supervision_loss = self.criterion(ampl_p, ampl_l)/(self.param.numOfSig*ampl_l.shape[0])
            self.log("supervision_loss", supervision_loss)
            loss += supervision_loss
        return loss,recons_loss
    def training_step(self, batch, batch_idx):
        if self.param.parameters["simulated"] is False:
            x = batch[0]
            ampl_batch = 0
        else:
            x, label = batch[0],batch[1]
            ampl_batch, alpha_batch = label[:, 0:-1], label[:, -1]

        # cond = random.randint(0,self.cond_max-1)
        dec_real, enct, enc,_,damp,_,mm,dec,decoded,b_spline_rec,spline_coeff = self(x)
        # self.param.parameters['spline_reg'] = cond
        # mu = enct[:, 0:self.param.numOfSig]
        # logvar = enct[:, self.param.numOfSig:2*(self.param.numOfSig)]
        mu = enct[:, 0:self.enc_out]
        logvar = enct[:, self.enc_out:2*(self.enc_out)]
        loss_mse,recons_loss = self.loss_function(dec, x, mu,logvar,mm,decoded,ampl_batch,enc,b_spline_rec,spline_coeff)
        self.training_step_outputs.append(loss_mse)
        self.log('damp',damp.mean())
        return {'loss': loss_mse,'recons_loss':recons_loss}

    def validation_step(self, batch, batch_idx):
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

        results = self.training_step(batch, batch_idx)
        try:
            if (self.current_epoch % self.param.parameters['val_freq'] == 0 and batch_idx == 0):
                # id = int(np.random.rand() * 300)
                id=150
                # sns.scatterplot(x=alpha_batch.cpu(), y=error.cpu())
                # sns.scatterplot(x=10*ampl_batch[:,12].cpu(),y=10*enc[:,12].cpu())
                # # plt.title(str(r2))
                # plt.show()
                # ampl_t = min_c + np.multiply(np.random.random(size=(1, 21)), (max_c - max_c))
                # y_n, y_wn = getSignal(ampl_t, 0, 5, 0, 0.5)
                rang = [1.8, 4]
                # id= 10
                # plotppm(np.fft.fftshift(np.fft.fft((y_n.T)).T), 0, 5,False, linewidth=0.3, linestyle='-')
                p1 = int(self.param.ppm2p(rang[0], (self.param.y_test_trun.shape[1])))
                p2 = int(self.param.ppm2p(rang[1], (self.param.y_test_trun.shape[1])))
                self.param.plotppm(np.fft.fftshift(np.fft.fft((self.param.y_test_trun[id, :])).T)[p2:p1], rang[0], rang[1], False, linewidth=0.3, linestyle='-',label="y_test")
                # cond_ = random.randint(0,self.cond_max-1)
                # self.param.parameters['spline_reg'] = cond_
                # plt.plot(np.fft.fftshift(np.fft.fft(np.conj(y_trun[id, :])).T)[250:450], linewidth=0.3)
                rec_signal,_,enc, fr, damp, ph,mm_v,_,decoded, spline_rec,spline_coeff = self(torch.unsqueeze(self.param.y_test_trun[id, :], 0).cuda())
                spline_loss = torch.linalg.vector_norm(spline_coeff[:, :-1] - spline_coeff[:, 1:])  #
                # self.log(f"spline_loss_val_{cond_}", spline_loss)
                # plotppm(np.fft.fftshift(np.fft.fft(((rec_signal).cpu().detach().numpy()[0,0:truncSigLen])).T), 0, 5,False, linewidth=1, linestyle='--')
                if self.param.parameters["decode"] == True:
                    if self.param.in_shape == 'real':
                        decoded = decoded[:,0,:]
                        self.param.plotppm(40+np.fft.fftshift(np.fft.rfft(
                            (decoded.cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T)[p2:p1], rang[0], rang[1],
                                True, linewidth=1, linestyle='--',label="decoded")
                    else:
                        decoded = decoded[:,0,:] + decoded[:,1,:]*1j
                        self.param.plotppm(40+np.fft.fftshift(np.fft.fft(
                            (decoded.cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T)[p2:p1], rang[0], rang[1],
                                True, linewidth=1, linestyle='--',label="decoded")
                rec_sig = rec_signal.cpu().detach().numpy()[0, 0:self.param.truncSigLen]


                self.param.plotppm(np.fft.fftshift(np.fft.fft(
                    (rec_sig)).T)[p2:p1], rang[0], rang[1],
                        False, linewidth=1, linestyle='--',label="rec_sig")
                plt.title("#Epoch: " + str(self.current_epoch))
                plt.legend()
                self.param.savefig(self.param.epoch_dir+"decoded_paper1_1_epoch_" + "_"+ str(self.tr_wei))
                # self.param.savefig(
                #     self.param.epoch_dir + "decoded_paper1_1_epoch_" + str(self.current_epoch) + "_" + str(self.tr_wei))

                if self.param.MM == True:
                    self.param.plotppm(5+10*np.fft.fftshift(np.fft.fft(((mm_v).cpu().detach().numpy()[0, 0:self.param.truncSigLen])).T)[p2:p1], rang[0], rang[1], False, linewidth=1,linestyle='--',label="mm_v")

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
                self.param.plot_basis(10*(enc).cpu().detach().numpy(), fr.cpu().detach().numpy(), damp.cpu().detach().numpy(), ph.cpu().detach().numpy(),rng=rang)
                # plt.plot(np.fft.fftshift(np.fft.fft(np.conj(rec_signal.cpu().detach().numpy()[0,0:trunc])).T)[250:450], linewidth=1,linestyle='--')
                plt.title("#Epoch: " + str(self.current_epoch))
                plt.legend()
                # self.param.savefig(self.param.epoch_dir+"fit_paper1_1_epoch_" + str(self.current_epoch) +"_"+ str(self.tr_wei))
                self.param.savefig(
                    self.param.epoch_dir + "fit_paper1_1_epoch_" + "_" + str(self.tr_wei))

                if self.param.parameters['spline']:
                    # rec_signal, _, enc, fr, damp, ph, mm_v, _, decoded, spline_rec, spline_coeff = self(
                    #     torch.unsqueeze(self.param.y_test_trun[id, :], 0).cuda(), cond_)
                    fig = plt.figure()
                    spline_rec = spline_rec.cpu().detach()
                    plt.plot(spline_rec.T)
                    # self.param.savefig(self.param.epoch_dir+"spline")
                    y_test = (self.param.y_test_trun[id, :]).unsqueeze(0)
                    y_test = self.param.zero_fill_torch(y_test, 1, self.param.parameters['zero_fill'][1]).cpu().detach()
                    plt.plot(torch.fft.fftshift(torch.fft.fft((y_test)).T)[self.p1:self.p2])
                    rec_sig = self.param.zero_fill_torch(rec_signal, 1, self.param.parameters['zero_fill'][1]).cpu().detach()
                    plt.plot(torch.fft.fftshift(torch.fft.fft((rec_sig)).T)[self.p1:self.p2] + spline_rec.cpu().detach().T)
                    # plt.title(f'condition:{cond_}')
                    tensorboard = self.logger.experiment
                    tensorboard.add_figure("recons", fig, self.current_epoch)
                    self.param.savefig(self.param.epoch_dir + "result")
        except:
            print("problem in plottuimg during validation")
        self.log("val_acc", results['loss'])
        self.log("val_recons_loss", results['recons_loss'])
        self.validation_step_outputs.append(results['loss'])
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
        self.log("epoch_los",avg_loss)
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
                    self.log(name,r2[idx])
                for name in self.selected_met:
                    performance+=r2[self.param.met_name.index(name)]
                performance=performance/len(self.selected_met)
                self.log("performance",performance)
                r2_total = torch.mean(r2)
                self.log("r2_total",r2_total)
                corr_total = torch.mean(corr)
                self.log("corr_total",corr_total)
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

    def crlb(self,x,noise_sd,ampl=None,percent = True, cal_met=True):
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

    def bspline(self,coeff):
        # cond = self.param.parameters['spline_reg']
        # coeff = self.embed[cond](coeff)
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

