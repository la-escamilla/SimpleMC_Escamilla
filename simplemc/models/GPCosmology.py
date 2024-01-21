import math as N
import numpy as np
from simplemc.models.LCDMCosmology import LCDMCosmology
from scipy.integrate import quad
from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from simplemc.cosmo.Parameter import Parameter

import matplotlib.pyplot as plt

class GPCosmology(LCDMCosmology):
    def __init__(self):

        self.Nbins_eos = 4
        mean_eos = -1
        self.params = [Parameter("zbin_eos%d"%i, mean_eos, 0.2, (-3.5, 0), "zbin_eos%d"%i) for i in range(self.Nbins_eos)]
        self.pvals = [i.value for i in self.params]
        self.z_i = np.linspace(0.0, 3.0, self.Nbins_eos)

        self.zinter = np.linspace(0.0, 3.0, 50)

        LCDMCosmology.__init__(self, mnu=0)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        l+= self.params
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            for i in range(self.Nbins_eos):
                if p.name == ("zbin_eos"+str(i)):
                    self.pvals[i] = p.value

        self.initialize()
        return True


    
    def de_eos(self, z):
        w_i = np.asarray(self.pvals)
        z_i = np.atleast_2d(np.linspace(0.0,3.0,len(self.params))).T
        kernel = RBF(1,(1e-2,1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, optimizer=None)
        gp.fit(z_i, w_i)
        w = gp.predict([[z]])
        return w


    def de_rhow(self, z):
        eos = self.de_eos(z)
        resultado = quad(lambda b: 3.0*(1.0+ eos)/(1.0+b), 0.0, z )
        return resultado[0]


    def initialize(self):
        rhow = [self.de_rhow(z) for z in self.zinter]
        self.rhow_inter = interp1d(self.zinter, rhow)
        return True


    ## this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self,a):
        z= 1./a - 1
        if z>= 3.0:
            rhow = (1.0-self.Om)
        else:
            rhow = (1.0-self.Om)*np.exp(self.rhow_inter(z))
        return self.Ocb/a**3 + self.Omrad/a**4 + rhow


