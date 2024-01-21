
import math as N
import numpy as np
from simplemc.models.LCDMCosmology import LCDMCosmology
from scipy.integrate import quad
from scipy.interpolate import interp1d
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import w1_par, w2_par, b_oscil_par

import matplotlib.pyplot as plt

class OscillatingEOS(LCDMCosmology):
    def __init__(self, varyw1=True, varyw2=True, varyb_oscil=True, model = 'model1'):
        self.varyw1  = varyw1
        self.varyw2  = varyw2
        self.varyb_oscil = varyb_oscil
        self.model = model

        self.w1 = w1_par.value
        self.w2 = w2_par.value
        self.b_oscil = b_oscil_par.value
        self.zinter = np.linspace(0.0, 3.0, 50)

        LCDMCosmology.__init__(self)
        self.updateParams([])

    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if (self.varyw1):  l.append(w1_par)
        if (self.varyw2):  l.append(w2_par)
        if (self.varyb_oscil): l.append(b_oscil_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "w1":
                self.w1 = p.value
            elif  p.name == "w2":
                self.w2 = p.value
            elif p.name == "b_oscil":
                self.b_oscil = p.value
        self.initialize()
        return True


     
    def de_eos(self, z):
        if self.model == 'model1':
            w = self.w1 + self.b_oscil*(1. - np.cos(np.log(1+z)))
        elif self.model == 'model2':
            w = self.w1 + self.b_oscil*(np.sin(np.log(1+z)))
        elif self.model == 'model3':
            w = self.w1 + self.b_oscil*((np.sin(1+z))/(1+z)-np.sin(1))
        elif self.model == 'model4':
            w = self.w1 + self.b_oscil*(z/(1+z))*(np.cos(1+z))
        elif self.model == 'model5': #el w1 es igual al b del overleaf
            w = -1 + ((self.b)/(1+z**2))*np.sin(self.w2*z)
        elif self.model == 'model6':
            w = -1 + ((self.b*z)/(1+z**2))*np.sin(self.w2*z)
        elif self.model == 'model7':
            w = -1 + ((self.b*z)/(1+z**2))*np.cos(self.w2*z)
        elif self.model == 'model8':
            w = -1 + ((self.b*z)/(1+z**2))*(np.cos(self.w2*z))**2
        elif self.model == 'model9':
            w =  -1 + ((self.b*z)/(1+z**2))*(np.cos(self.w2*z))**3
        return w


    def de_rhow(self, z):
        eos = self.de_eos(z)
        resultado = quad(lambda b: 3.0*(1.0+ eos)/(1.0+b), 0.0, z )
        return resultado[0]
   

    def initialize(self):   
        #w_inter = [self.de_eos(z) for z in self.zinter]
        rhow = [self.de_rhow(z) for z in self.zinter]
        self.rhow_inter = interp1d(self.zinter, rhow)

        #plt.plot(self.zinter, rhow)
        #plt.show()
        return True

    
    ## this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self,a):
        z= 1./a - 1
        if z>= 3.0:
            rhow = (1.0-self.Om)
        else:
            rhow = (1.0-self.Om)*np.exp(self.rhow_inter(z)) 
        #return (self.Ocb/a**3+self.Omrad/a**4 +(1.0-self.Om)*(np.exp(self.luisfunction(z)[1]))  )
        return self.Ocb/a**3 + self.Omrad/a**4 + rhow

