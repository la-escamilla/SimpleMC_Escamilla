import math as N
import numpy as np
from simplemc.models.LCDMCosmology import LCDMCosmology
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import w_ide_par, alpha_ssik_par, sigma_ssik_par

class IDESignSwitchCosmology(LCDMCosmology):
    def __init__(self, varyw_ide=True, varyalpha_ssik=True, varysigma_ssik=True):

        #self.Nbins_eos = 5
        #mean_eos = -1
        #self.params = [Parameter("zbin_eos%d"%i, mean_eos, 0.2, (-3.5, 0), "zbin_eos%d"%i) for i in range(self.Nbins_eos)]
        #self.pvals = [i.value for i in self.params]
        #self.z_i = np.linspace(0.0, 3.0, self.Nbins_eos)


        self.varyw_ide = varyw_ide
        self.w_ide = w_ide_par.value
        self.varyalpha_ssik = varyalpha_ssik
        self.alpha_ssik = alpha_ssik_par.value
        self.varysigma_ssik = varysigma_ssik
        self.sigma_ssik = sigma_ssik_par.value

        self.zinter = np.linspace(0.0, 3.0, 50)

        LCDMCosmology.__init__(self, mnu=0)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        #l+= self.params
        if (self.varyw_ide): l.append(w_ide_par)
        if (self.varyalpha_ssik): l.append(alpha_ssik_par)
        if (self.varysigma_ssik): l.append(sigma_ssik_par)
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == ("w_ide"):
                self.w_ide = p.value
            elif p.name == ("alpha_ssik"):
                self.alpha_ssik = p.value
            elif p.name == ("sigma_ssik"):
                self.sigma_ssik = p.value
            #for i in range(self.Nbins_eos):
            #    if p.name == ("zbin_eos"+str(i)):
            #        self.pvals[i] = p.value
            

        self.initialize()
        return True

    def bines(self, w_2, w_1, z_2, z_1, eta):
        return (w_2-w_1)*(1.0+np.tanh((z_2-z_1)/eta))/2.0

    #def de_eos(self, z):
    #    w = self.pvals[0]
    #    for jj in range(self.Nbins_eos - 1):
    #        w+=self.bines(self.pvals[jj+1], self.pvals[jj], z, self.z_i[jj+1], eta=0.15)
    #    return w
    
    def de_dm_rhow(self, func, z):
        #sigma = 0.001
        alpha = N.floor(self.alpha_ssik)
        rho_de = func[0]
        rho_dm = func[1]
        drhodedz = (3.0/(1.0+z))*((1+self.w_ide+self.sigma_ssik)*rho_de-self.sigma_ssik*rho_dm*alpha)
        drhodmdz = (3.0/(1.0+z))*((1+self.sigma_ssik*alpha)*rho_dm-self.sigma_ssik*rho_de)
        dfuncdz = [drhodedz, drhodmdz]
        return dfuncdz

    def initialize(self):
        rhowde = np.reshape(odeint(self.de_dm_rhow ,[1.0-self.Om,self.Om] ,self.zinter)[:,0],len(self.zinter))
        rhowdm = np.reshape(odeint(self.de_dm_rhow ,[1.0-self.Om,self.Om] ,self.zinter)[:,1],len(self.zinter))
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
            return om_dm + self.Omrad/a**4 + om_de
        elif om_dm + self.Omrad/a**4 + om_de <=0 :
            return 0.5


