
import math as N
import numpy as np
from simplemc.models.LCDMCosmology import LCDMCosmology
from scipy.integrate import quad
from scipy.interpolate import interp1d
from simplemc.cosmo.Parameter import Parameter

import matplotlib.pyplot as plt

class WiggleCDMCosmology(LCDMCosmology):
    def __init__(self):


        self.Nbins_wiggle = 6
        mean_wiggle = 0.0
        self.params = [Parameter("zbin_wiggle%d"%i, mean_wiggle, 0.001, (-0.01, 0.01), "zbin_wiggle%d"%i) for i in range(self.Nbins_wiggle)]
        self.pvals = [i.value for i in self.params]
        self.z_i = np.linspace(0.0, 3.0, self.Nbins_wiggle+1)

        self.zinter = np.linspace(0.0, 3.0, 50)

        LCDMCosmology.__init__(self, mnu=0)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        #l.append(zbin_rho_par)
        l+= self.params
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            for i in range(self.Nbins_wiggle):
                if p.name == ("zbin_wiggle"+str(i)):
                    self.pvals[i] = p.value

        self.initialize()
        return True

     
    def bines(self, w_2, w_1, z_2, z_1, eta):
        return (w_2-w_1)*(1.0+np.tanh((z_2-z_1)/eta))/2.0

    def de_eos(self, z):
        w = self.pvals[0]
        for jj in range(self.Nbins_wiggle - 1):
            w+=self.bines(self.pvals[jj+1], self.pvals[jj], z, self.z_i[jj+1], eta=0.15)
        return w
   

    def initialize(self):
        #w_inter = [self.de_eos(z) for z in self.zinter]
        rhow = [self.de_eos(z) for z in self.zinter]
        self.rhow_inter = interp1d(self.zinter, rhow)

        #plt.plot(self.zinter, rhow)
        #plt.show()
        return True

    

    ## this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self,a):
        z= 1./a - 1
        
        rhow = self.h*100.0*((self.Om*(1+z)**3+1.0-self.Om)**0.5) 
        #return (self.Ocb/a**3+self.Omrad/a**4 +(1.0-self.Om)*(np.exp(self.luisfunction(z)[1]))  )
        return ((rhow)/(self.h*100*(1+self.rhow_inter(z)*rhow)))**2

