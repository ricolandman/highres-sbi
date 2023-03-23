#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import astropy.units as u
import wandb
#from specutil import *
import astropy.units as u
import astropy.constants as const
from PyAstronomy.pyasl import fastRotBroad

from dawgz import job, schedule
from itertools import islice
from pathlib import Path
from torch import Tensor
from tqdm import tqdm

from lampe.data import H5Dataset
from lampe.inference import NPE, NPELoss
from lampe.nn import ResMLP
from lampe.utils import GDStep

from zuko.flows import NAF
from zuko.distributions import BoxUniform
from generate import param_set
from parameter import *

# from ees import Simulator, LOWER, UPPER


scratch = os.environ.get('SCRATCH', '')
# scratch = '/users/ricolandman/Research_data/npe_crires/'
datapath = Path(scratch) / 'highres-sbi'
savepath = Path(scratch) / 'highres-sbi/runs'


class SoftClip(nn.Module):
    def __init__(self, bound: float = 1.0):
        super().__init__()

        self.bound = bound

    def forward(self, x: Tensor) -> Tensor:
        return x / (1 + abs(x / self.bound))


class CNN(nn.Module):
    def __init__(self, n_channels_in=2, n_channels_out=128):
        super(CNN, self).__init__()
        
        ## Convolutional layers ##
        self.layer1 = self.ConvLayer(n_channels_in, 4, ksize_conv=8, strd_conv=4)
        self.layer2 = self.ConvLayer(4, 8, ksize_conv=8, strd_conv=4)
        self.layer3 = self.ConvLayer(8, 16, ksize_conv=8, strd_conv=4)
        self.layer4 = self.ConvLayer(16, 32, ksize_conv=8, strd_conv=4)
        
        ## Fully connected layers ##
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1504, 256),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(128, n_channels_out))
        self.fc = self.fc.to(torch.float32)
        
    def ConvLayer(self, nb_neurons_in, nb_neurons_out, ksize_conv=3, strd_conv=1, pad_conv=0, ksize_pool=3, strd_pool=1, pad_pool=0):
        '''
        Define a convolutional layer
        '''
        layer = nn.Sequential(
            nn.Conv1d(nb_neurons_in, nb_neurons_out, 
                      kernel_size=ksize_conv, stride=strd_conv, padding=pad_conv),
            #nn.BatchNorm1d(nb_neurons_out),
            nn.ELU())
            #nn.MaxPool1d(kernel_size=ksize_pool, stride=strd_pool, padding=pad_pool))
        return layer

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #print(out.dtype)
        out = out.view(out.size(0), -1)#.to(torch.float32)  # Flatten for fully connected layers
        out = self.fc(out)
        out = out.to(torch.double)
        
        return out

class NPEWithEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = CNN(2, 128)
        self.npe = NPE(
            19, 128,
            #moments=((l + u) / 2, (u - l) / 2),
            transforms=3,
            build=NAF,
            hidden_features=[512] * 5,
            activation=nn.ELU,
        ).to(torch.float64)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        y = self.embedding(x)
        if np.any(np.isnan(y.detach().cpu().numpy())):
             print('NaNs in embedding')
        return self.npe(theta.to(torch.double), y)

    def flow(self, x: Tensor):  # -> Distribution
        out = self.npe.flow(self.embedding(x))
        if np.any(np.isnan(out.detach().cpu().numpy())):
             print('NaNs in flow')
        return out
    

class BNPELoss(nn.Module):
    def __init__(self, estimator, prior, lmbda=100.0):
        super().__init__()
        self.estimator = estimator
        self.prior = prior
        self.lmbda = lmbda
    def forward(self, theta, x):
        theta_prime = torch.roll(theta, 1, dims=0)
        log_p, log_p_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )
        l0 = -log_p.mean()
        lb = (torch.sigmoid(log_p - self.prior.log_prob(theta)) + torch.sigmoid(log_p_prime - self.prior.log_prob(theta_prime)) - 1).mean().square()
        return l0 + self.lmbda * lb

    

class Data():
    def __init__(self, path= 'data_to_fit.dat'):
        self.path = Path(path)
        self.data = np.loadtxt(self.path)
        self.wl, f, er, _, trans = self.data.T
        self.flux, self.err = self.FluxandError_processing(f, er)
        self.model_wavelengths = self.get_modelW()
        self.data_wavelengths = self.wl/1000
        self.data_wavelengths_norm = self.norm_data_wavelengths()
                
    def unit_conversion(self, flux, distance=4.866*u.pc):
        flux_units = flux * u.erg/u.s/u.cm**2/u.nm
        flux_dens_emit = (flux_units * distance**2/const.R_jup**2).to(u.W/u.m**2/u.micron)
        return flux_dens_emit.value
        
    def FluxandError_processing(self, flux, err):
        nans = np.isnan(flux)
        flux[nans] = np.interp(self.wl[nans], self.wl[~nans], flux[~nans])
        flux = self.unit_conversion(flux)
        flux_scaling = 1./np.nanmean(flux)
        
        err[nans] = np.interp(self.wl[nans], self.wl[~nans], err[~nans])
        err = self.unit_conversion(err)
                
        return flux_scaling, err 
        
    def get_modelW(self):
        sim_res = 2e5
        dlam = 2.350/sim_res
        return np.arange(2.320, 2.371, dlam)
        
    def norm_data_wavelengths(self):
        return (self.data_wavelengths - np.nanmean(self.data_wavelengths))/\
                    (np.nanmax(self.data_wavelengths)-np.nanmin(self.data_wavelengths))
        
#     def main(self):

class Processing():
    d = Data()
    data_wavelengths = d.data_wavelengths
    model_wavelengths = d.model_wavelengths
    flux_scaling = d.flux
    data_wavelengths_norm = d.data_wavelengths_norm
    
    def __call__(self, theta, x):
        self.theta = theta
        self.x = x
        self.param_set_ext, self.theta_ext = self.params_ext()  #external param set, one batch of theta ext
        self.x_new = self.process_x()
        self.theta_new = self.params_combine()
        
        return self.theta_new, self.x_new
    
    def params_ext(self):
        batch_size = self.theta.shape[0]
        # Define additional parameters
        radius = Parameter('radius', uniform_prior(0.8, 2.0))
        rv = Parameter('rv', uniform_prior(10, 30))
        limb_dark = Parameter('limb_dark', uniform_prior(0,1))
        vsini = Parameter('vsini', uniform_prior(0, 50))
        param_set_ext = ParameterSet([radius, rv, vsini, limb_dark])
        # Generate theta_ext
        theta_ext = param_set_ext.sample(batch_size)
        return param_set_ext, theta_ext
        
    def process_x(self):
        batch_size = self.theta.shape[0]
        x_obs = np.zeros((batch_size, 2, self.data_wavelengths.size))
        for i, xi, theta_ext_i in zip(range(self.x.shape[0]), self.x, self.theta_ext):
            param_dict = self.param_set_ext.param_dict(theta_ext_i)
            #Apply radius scaling
            xi = xi * param_dict['radius']**2
            # Apply line spread function and radial velocity
            xi = fastRotBroad(self.model_wavelengths,xi, param_dict['limb_dark'], param_dict['vsini'])
            shifted_wavelengths = (1+param_dict['rv']/const.c.to(u.km/u.s).value) * self.model_wavelengths
            # Convolve to instrument resolution
            x_obs[i, 0, :] = np.interp(self.data_wavelengths, shifted_wavelengths, xi)
        # Scaling
        x_obs[:,0] = x_obs[:,0] * self.flux_scaling
        x_obs[:,1, :] = self.data_wavelengths_norm
#       if np.any(np.isnan(x_obs)):
#       print('NaNs in x_obs') 
        return x_obs

        
    def params_combine(self):
        # Add theta_ext to theta's
        ## add to param_set here
        theta = self.theta.numpy()
        theta_norm = (self.theta-param_set.lower)/(param_set.upper - param_set.lower)
        theta_ext_norm = (self.theta_ext - self.param_set_ext.lower)/(self.param_set_ext.upper - self.param_set_ext.lower)
        theta_new = np.concatenate([theta_norm, theta_ext_norm], axis=-1)
        if np.any(np.isnan(theta)):
             print('NaNs in theta')

        return theta_new
                
                    
def noisy(data) -> np.ndarray:
    data_uncertainty = Data().err
    theta, x = data
    x[:,0] = x[:,0] + data_uncertainty * np.random.randn(*x[:,0].shape)
    return theta, x
   

def train(): 

    config_dict = {
        
                'embedding': 'shallow',  #shallow = [2,3,5], deep = [3,5,7]
                'flow': 'NAF',
                'transforms': 3, 
                'hidden_features': 512, # hidden layers of the autoregression network
                'activation': 'ELU',
                'optimizer': 'AdamW',
                'init_lr': 1e-3,
                'weight_decay': 1e-2,
                'scheduler': 'ReduceLROnPlateau',
                'min_lr': 1e-6,
                'patience': 32,
                'epochs': 1024,
                'stop_criterion': 'early', 
                'batch_size': 2048,
                'gradient_steps_train': 1024, 
                'gradient_steps_valid': 256
             } 

    # Run
#     run = wandb.init(project='highres-ear')

    # Data
    trainset = H5Dataset(datapath / 'train.h5', batch_size=128, shuffle=True)
    validset = H5Dataset(datapath / 'valid.h5', batch_size=128, shuffle=True)

    # Training
    process = Processing()
    estimator = NPEWithEmbedding().cuda()
    prior = BoxUniform(torch.tensor(param_set.lower).cuda(), torch.tensor(param_set.upper).cuda())
    loss = NPELoss(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=1e-4, weight_decay=1e-2)
    step = GDStep(optimizer, clip=1.0)
    scheduler = sched.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        min_lr=1e-6,
        patience=32,
        threshold=1e-2,
        threshold_mode='abs',
    )

    def pipe(theta: Tensor, x: Tensor) -> Tensor:
        theta, x = noisy(process(theta,x))
        theta, x = torch.from_numpy(theta).cuda(), torch.from_numpy(x).cuda()
        return loss(theta, x)

    for epoch in tqdm(range(1024), unit='epoch'):
        estimator.train()
        start = time.time()

        losses = torch.stack([
            step(pipe(theta, x))
            for theta, x in islice(trainset, 1024)
        ]).cpu().numpy()
        

        end = time.time()
        estimator.eval()

        with torch.no_grad():
            losses_val = torch.stack([
                pipe(theta, x)
                for theta, x in islice(validset, 256)
            ]).cpu().numpy()

#         run.log({
#             'lr': optimizer.param_groups[0]['lr'],
#             'loss': np.nanmean(losses),
#             'loss_val': np.nanmean(losses_val),
#             'nans': np.isnan(losses).mean(),
#             'nans_val': np.isnan(losses_val).mean(),
#             'speed': len(losses) / (end - start),
#         })

        scheduler.step(np.nanmean(losses_val))

        if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            break

    runpath = savepath / run.name
    runpath.mkdir(parents=True, exist_ok=True)

    if epoch % 50 ==0 : 
            torch.save({
            'estimator': estimator.state_dict(),
            'optimizer': optimizer.state_dict(),
        },  runpath / f'states_{epoch}.pth')

#     run.finish()


train()
