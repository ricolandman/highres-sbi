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
from generate import param_set
from parameter import *

#from ees import Simulator, LOWER, UPPER


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
        if np.any(np.isnan(y.detach().numpy())):
             print('NaNs in embedding')
        return self.npe(theta.to(torch.double), y)

    def flow(self, x: Tensor):  # -> Distribution
        out = self.npe.flow(self.embedding(x))
        if np.any(np.isnan(out.detach().numpy())):
             print('NaNs in flow')
        return out


#@job(array=3, cpus=2, gpus=1, ram='8GB', time='1-00:00:00')
def train(model_wavelengths, data_wavelengths, data_uncertainty, data_scaling):
    # Run
    run = wandb.init(project='ear')
    wl_normalized = (data_wavelengths - np.nanmean(data_wavelengths))/\
                    (np.nanmax(data_wavelengths)-np.nanmin(data_wavelengths))
    
    # Define additional parameters
    radius = Parameter('radius', uniform_prior(0.8, 2.0))
    rv = Parameter('rv', uniform_prior(10, 30))
    limb_dark = Parameter('limb_dark', uniform_prior(0,1))
    vsini = Parameter('vsini', uniform_prior(0, 50))
    param_set_ext = ParameterSet([radius, rv, vsini, limb_dark])
    
    def noisy(theta, x) -> np.ndarray:
        batch_size = theta.shape[0]

        # Generate theta_ext
        theta_ext = param_set_ext.sample(batch_size)

        x_obs = np.zeros((batch_size, 2, data_wavelengths.size))
        for i, xi, theta_ext_i in zip(range(x.shape[0]), x, theta_ext):
            param_dict = param_set_ext.param_dict(theta_ext_i)
            #Apply radius scaling
            xi = xi * param_dict['radius']**2

            # Apply line spread function and radial velocity
            xi = fastRotBroad(model_wavelengths,xi, param_dict['limb_dark'], param_dict['vsini'])
            shifted_wavelengths = (1+param_dict['rv']/const.c.to(u.km/u.s).value) * model_wavelengths

            # Convolve to instrument resolution
            #xi = Spectrum(xi, shifted_wavelengths)
            #xi = convolve_to_resolution(xi, 100_000, 200_000)
            #x_obs[i, 0, :] = xi.at(data_wavelengths)
            x_obs[i, 0, :] = np.interp(data_wavelengths, shifted_wavelengths, xi)

        # Add noise
        x_obs[:,0] = x_obs[:,0] + data_uncertainty * np.random.randn(*x_obs[:,0].shape)
        #x_obs = np.maximum(x_obs, 0)

        # Scaling
        x_obs[:,0] = x_obs[:,0] * flux_scaling
        x_obs[:,1, :] = wl_normalized

        # Add theta_ext to theta's
        theta = theta.numpy()
        theta_norm = (theta-param_set.lower)/(param_set.upper - param_set.lower)
        theta_ext_norm = (theta_ext - param_set_ext.lower)/(param_set_ext.upper - param_set_ext.lower)
        theta = np.concatenate([theta_norm, theta_ext_norm], axis=-1)
        if np.any(np.isnan(theta)):
             print('NaNs in theta')
        if np.any(np.isnan(x_obs)):
             print('NaNs in x_obs')
        return theta, x_obs

    # Data
    trainset = H5Dataset(datapath / 'train.h5', batch_size=128, shuffle=True)
    #trainset = H5Dataset(datapath / 'train.h5', batch_size=2048, shuffle=True)
    validset = H5Dataset(datapath / 'valid.h5', batch_size=128, shuffle=True)

    # Training
    estimator = NPEWithEmbedding()#.cuda()
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
        theta, x = noisy(theta, x)
        #theta, x = theta.cuda(), x.cuda()
        theta, x = torch.from_numpy(theta), torch.from_numpy(x)
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

        run.log({
            'lr': optimizer.param_groups[0]['lr'],
            'loss': np.nanmean(losses),
            'loss_val': np.nanmean(losses_val),
            'nans': np.isnan(losses).mean(),
            'nans_val': np.isnan(losses_val).mean(),
            'speed': len(losses) / (end - start),
        })

        scheduler.step(np.nanmean(losses_val))

        if optimizer.param_groups[0]['lr'] <= scheduler.min_lrs[0]:
            break

    runpath = savepath / run.name
    runpath.mkdir(parents=True, exist_ok=True)

    torch.save(estimator.state_dict(), runpath / 'state.pth')

    run.finish()

def unit_conversion(flux, distance=4.866*u.pc):
    flux_units = flux * u.erg/u.s/u.cm**2/u.nm
    flux_dens_emit = (flux_units * distance**2/const.R_jup**2).to(u.W/u.m**2/u.micron)
    return flux_dens_emit.value

if __name__ == '__main__':
    x = np.loadtxt('data_to_fit.dat')
    wl, flux, err, _, trans = x.T
    nans = np.isnan(flux)
    err[nans] = np.interp(wl[nans], wl[~nans], err[~nans])
    flux[nans] = np.interp(wl[nans], wl[~nans], flux[~nans])
    flux = unit_conversion(flux)
    err = unit_conversion(err)

    flux_scaling = 1./np.nanmean(flux)

    sim_res = 2e5
    dlam = 2.350/sim_res
    model_wavelengths = np.arange(2.320, 2.371, dlam)
    
    train(model_wavelengths, wl/1000, err, flux_scaling)
    '''
    schedule(
        train,
        name='Training',
        backend='slurm',
        env=[
            'source ~/.bashrc',
            'conda activate ear',
            'export WANDB_SILENT=true',
        ]
    )'''
