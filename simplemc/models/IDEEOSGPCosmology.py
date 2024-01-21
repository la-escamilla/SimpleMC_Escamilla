
import math as N
import numpy as np
from simplemc.models.LCDMCosmology import LCDMCosmology
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from simplemc.cosmo.Parameter import Parameter

class IDEEOSGPCosmology(LCDMCosmology):
    def __init__(self):

        self.Nbins_ide = 5
        mean_ide = 0.0
        self.params_ide = [Parameter("zbin_ide%d"%i, mean_ide, 0.005, (-1.0, 1.0), "zbin_ide%d"%i) for i in range(self.Nbins_ide)]
        self.pvals_ide = [i.value for i in self.params_ide]


        
        self.Nbins_eos = 5
        mean_eos = -1.0
        self.params_eos = [Parameter("zbin_eos%d"%i, mean_eos, 1.0, (-5.0, 1.), "zbin_eos%d"%i) for i in range(self.Nbins_eos)]
        self.pvals_eos = [i.value for i in self.params_eos]


        self.zinter = np.linspace(0.0, 3.0, 50)

        LCDMCosmology.__init__(self, mnu=0)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        l+= self.params_eos
        l+= self.params_ide
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            for j in range(self.Nbins_eos):
                if p.name == ("zbin_eos"+str(j)):
                    self.pvals_eos[j] = p.value
            for i in range(self.Nbins_ide):
                if p.name == ("zbin_ide"+str(i)):
                    self.pvals_ide[i] = p.value
        self.initialize()
        return True


    def de_ide(self, z):
        ide_i = np.asarray(self.pvals_ide)
        z_i = np.atleast_2d(np.linspace(0.0,3.0,len(self.params_ide))).T
        kernel = RBF(1,(1e-2,1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, optimizer=None)
        gp.fit(z_i, ide_i)
        ide = gp.predict([[z]])
        return ide[0]/1000.

    def de_eos(self, z):
        w_i = np.asarray(self.pvals_eos)
        z_i = np.atleast_2d(np.linspace(0.0,3.0,len(self.params_eos))).T
        kernel = RBF(1,(1e-2,1e2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, optimizer=None)
        gp.fit(z_i, w_i)
        w = gp.predict([[z]])
        return w[0]

    
    def de_rhow(self, rho_de, z):
        drhodedz = (3.0/(1.0+z))*((1.0+self.de_eos(z))*rho_de+self.de_ide(z))
        return drhodedz

    def dm_rhow(self, rho_dm, z):
        drhodmdz = (3.0/(1.0+z))*(rho_dm-self.de_ide(z))
        return drhodmdz

    def initialize(self):
        rhowde = np.reshape(odeint(self.de_rhow ,1.0-self.Om ,self.zinter),len(self.zinter))
        rhowdm = np.reshape(odeint(self.dm_rhow ,self.Om ,self.zinter),len(self.zinter))
        self.rhowde_inter = interp1d(self.zinter, rhowde)
        self.rhowdm_inter = interp1d(self.zinter, rhowdm)
        return True

    ## this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self,a):
        z= 1./a - 1
        if z>= 3.0:
            om_de = (1.0-self.Om)
            om_dm = (self.Om)
        else:
            om_de = self.rhowde_inter(z)
            om_dm = self.rhowdm_inter(z)
        #return om_dm + self.Omrad/a**4 + om_de
        if om_dm+self.Omrad/a**4 + om_de > 0 :
            return om_dm + self.Omrad/a**4 + om_de
        elif om_dm + self.Omrad/a**4 + om_de <=0 :
            return 0.5


