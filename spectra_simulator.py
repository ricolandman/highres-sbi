import numpy as np
import matplotlib.pyplot as plt
import os
from specutil import *
from scipy.interpolate import CubicSpline
from petitRADTRANS import Radtrans
from petitRADTRANS.poor_mans_nonequ_chem import interpol_abundances
from PyAstronomy.pyasl import fastRotBroad
from cloud_cond import return_cloud_mass_fraction, simple_cdf_MgSiO3, simple_cdf_Fe
from PyAstronomy.pyasl import fastRotBroad
from petitRADTRANS.physics import PT_ret_model



def abundances(press, temp, feh, C_O, P_quench=None):
    COs = np.ones_like(press)*C_O
    fehs = np.ones_like(press)*feh
    mass_fractions = interpol_abundances(COs,fehs, temp,press, P_quench)
    return mass_fractions

def make_pt(params, pressures):
    temp_arr = np.array([params['T3'],params['T2'],params['T1']])

    #delta = 10**params['log_delta']
    delta = (10.0**params['log_delta'])**(-params['alpha'])
    #delta = (1e6 * 10 ** (-3 + 5 * params['log_delta'])) ** (-params['alpha'])
    temperatures = PT_ret_model(temp_arr, \
                            delta,
                            params['alpha'],
                            params['T_int'],
                            pressures,
                            params['FEH'],
                            params['C_O'],
                            conv=True)
    return temperatures

class SpectrumMaker():
    def __init__(self, wavelengths, param_set,
            spectral_resolution=100_000, 
            species=['H2O_main_iso', 'CO_main_iso', 'CO_36'], 
            include_clouds=False,
            lbl_opacity_sampling=5, scat=True, pressures=None):
        self.species = species
        self.param_set = param_set
        self.spectral_resolution = spectral_resolution
        self.wavelengths = wavelengths
        if pressures is None:
            self.pressures = np.logspace(-6, 2, 80)
        else:
            self.pressures = pressures
        self.include_clouds = include_clouds
        wlen_range = np.array([np.min(self.wavelengths), np.max(self.wavelengths)])
        if include_clouds:
            self.atmosphere= Radtrans(line_species=species,rayleigh_species = ['H2', 'He'],
                                      continuum_opacities = ['H2-H2', 'H2-He'],
                                      wlen_bords_micron=wlen_range, mode='lbl',
                                      cloud_species=['MgSiO3(c)_cd','Fe(c)_cd'],
                                      do_scat_emis = scat,
                                       lbl_opacity_sampling = lbl_opacity_sampling)
        else:
            self.atmosphere= Radtrans(line_species=species,rayleigh_species = ['H2', 'He'],
                                      continuum_opacities = ['H2-H2', 'H2-He'],
                                      wlen_bords_micron=wlen_range, mode='lbl',
                                       lbl_opacity_sampling = lbl_opacity_sampling)

        self.atmosphere.setup_opa_structure(self.pressures)
        self.prt_press = self.atmosphere.press

    def __call__(self, param_arr):
        param_dict = self.param_set.param_dict(param_arr)
        return self.get_spectrum(param_dict)

    def get_spectrum(self, params):
        gravity = 10**params['log_g']
        if 'full_PT' in params:
            temperature = params['full_PT']
        else:
            temperature = make_pt(params, self.pressures)
        
        if 'log_Pquench' in params.keys():
            Pquench = 10**params['log_Pquench']
        else:
            Pquench = None
        abunds = abundances(self.pressures, temperature, params['FEH'], params['C_O'], Pquench)
        mass_fractions = self.get_abundance_dict(abunds, params['log_iso_rat'])
        MMW = abunds['MMW']
        if self.include_clouds:
            eq_MgSiO3 = return_XMgSiO3(params['FEH'], params['C_O'])
            P_base_MgSiO3 = simple_cdf_MgSiO3(self.pressures, temperature, params['FEH'], params['C_O'], np.mean(MMW))
            mass_fractions['MgSiO3(c)'] = np.zeros_like(temperature)
            mass_fractions['MgSiO3(c)'][self.pressures <= P_base_MgSiO3] = \
                              eq_MgSiO3 * 10**params['log_MgSiO3'] * \
                              (self.pressures[self.pressures <= P_base_MgSiO3]/P_base_MgSiO3)**params['fsed']
            eq_Fe = return_XFe(params['FEH'], params['C_O'])
            P_base_Fe= simple_cdf_Fe(self.pressures, temperature, params['FEH'], params['C_O'], np.mean(MMW))
            mass_fractions['Fe(c)'] = np.zeros_like(temperature)
            mass_fractions['Fe(c)'][self.pressures <= P_base_Fe] = \
                              eq_Fe * 10**params['log_Fe'] * \
                              (self.pressures[self.pressures <= P_base_Fe]/P_base_Fe)**params['fsed']

            Kzz = 10**params['log_Kzz']*np.ones_like(temperature)
        if self.include_clouds:
            self.atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW,
                                Kzz=Kzz,
                                fsed = params['fsed'], 
                                sigma_lnorm = params['sigma_lnorm'])

        else:
            self.atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW)
        
        wl = const.c.to(u.km/u.s).value/self.atmosphere.freq/1e-9
        flux = self.atmosphere.flux * u.erg/u.cm**2/u.s/u.Hz
        flux = flux *const.c/(wl*u.micron)**2
        flux = flux.to(u.W/u.m**2/u.micron)
        spec = Spectrum(flux, wl).at(self.wavelengths)
        '''
        waves_even = np.linspace(np.min(wl), np.max(wl), wl.size)
        spec = fastRotBroad(waves_even, spec.at(waves_even), params['limb_dark'], params['vsini'])
        shifted_wl = (1+params['rv']/const.c.to(u.km/u.s).value)*waves_even
        spec = Spectrum(spec, shifted_wl)
        spec = convolve_to_resolution(spec, self.spectral_resolution)
        '''
        return spec

    def get_abundance_dict(self, abunds, log_iso_rat=None):
        mass_fractions = {}
        for specie in self.species:
            if specie=='H2O_main_iso':
                mass_fractions[specie] = abunds['H2O']
            elif specie=='CO_main_iso':
                mass_fractions[specie] = abunds['CO']
            elif specie=='CH4_main_iso':
                mass_fractions[specie] = abunds['CH4']
            elif specie=='CO_36':
                mass_fractions[specie] = 10**(log_iso_rat)*abunds['CO']
        mass_fractions['H2'] = abunds['H2']
        mass_fractions['He'] = abunds['He']
        return mass_fractions
