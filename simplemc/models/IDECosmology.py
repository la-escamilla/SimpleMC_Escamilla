import math as N
import numpy as np
from simplemc.models.LCDMCosmology import LCDMCosmology
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import w_ide_par

class IDECosmology(LCDMCosmology):
    def __init__(self, varyw_ide=True):
 
        self.Nbins_ide = 5
        #Nbins_ide += 1
        mean_ide  = 0.0
        size_step = []
        iniide = []
        finide = []
        for ii in range(self.Nbins_ide):
            if np.linspace(0.0,3.0,self.Nbins_ide+1)[ii]<1.0:
                size_step += [0.2]
                iniide += [-2.0]
                finide += [4.0]
            elif np.linspace(0.0,3.0,self.Nbins_ide+1)[ii]>=1.0:
                size_step += [1.0]
                iniide += [-15.0]
                finide += [15.0]
        self.params_ide = [Parameter("zbin_ide%d"%i, mean_ide, size_step[i], (iniide[i], finide[i]), "zbin_ide%d"%i) for i in range(self.Nbins_ide)]

        #self.parvals = zbin_rho_par
        #self.zs = [i.value for i in self.parvals]
        #names   = [i.name  for i in self.parvals]
        #self.index = dict((j,i) for i,j in enumerate(names))



        #self.Nbins_ide = 5
        #mean_ide = 0
        #self.params_ide = [Parameter("zbin_ide%d"%i, mean_ide, 0.05, (-12.0, 2.0), "zbin_ide%d"%i) for i in range(self.Nbins_ide)]
        
        self.pvals_ide = [i.value for i in self.params_ide]
        self.z_i_ide = np.linspace(0.0, 3.0, self.Nbins_ide)
        
        self.varyw_ide = varyw_ide
        self.w_ide = w_ide_par.value
        
        self.zinter = np.linspace(0.0, 3.0, 50)

        LCDMCosmology.__init__(self, mnu=0)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        #l+= self.params
        if (self.varyw_ide): l.append(w_ide_par)
        l+= self.params_ide
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == ("w_ide"):
                self.w_ide = p.value
            #for i in range(self.Nbins_eos):
            #    if p.name == ("zbin_eos"+str(i)):
            #        self.pvals[i] = p.value
            for i in range(self.Nbins_ide):
                if p.name == ("zbin_ide"+str(i)):
                    self.pvals_ide[i] = p.value

        self.initialize()
        return True

    def bines(self, w_2, w_1, z_2, z_1, eta):
        return (w_2-w_1)*(1.0+np.tanh((z_2-z_1)/eta))/2.0

    #def de_eos(self, z):
    #    w = self.pvals[0]
    #    for jj in range(self.Nbins_eos - 1):
    #        w+=self.bines(self.pvals[jj+1], self.pvals[jj], z, self.z_i[jj+1], eta=0.15)
    #    return w
    
    def de_ide(self, z):
        ide = self.pvals_ide[0]
        for jj in range(self.Nbins_ide - 1):
            ide+=self.bines(self.pvals_ide[jj+1], self.pvals_ide[jj], z, self.z_i_ide[jj+1], eta=0.15)
        return ide/1.

    def de_rhow(self, rho_de, z):
        #drhodedz = (3.0/(1.0+z))*((1.0+self.de_eos(z))*rho_de+self.de_ide(z))
        drhodedz = (3.0/(1.0+z))*((1+self.w_ide)*rho_de+self.de_ide(z))
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
        if om_dm+self.Omrad/a**4 + om_de > 0 :
            return om_dm + self.Omrad/a**4 + om_de + self.Obh2/a**3
        elif om_dm + self.Omrad/a**4 + om_de <=0 :
            return 0.5


