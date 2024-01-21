from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.paramDefs import a_wave_par, b_wave_par, zcross_wave_par
import math as N
import numpy as np

#TODO Add more DE EoS for comparison



class HermitWaveletCDMCosmology(LCDMCosmology):
    
    def __init__(self, varya_wave=True, varyb_wave=True, varyzcross_wave=True, type_wavelet = 'hermitian_1'):
        self.varya_wave  = varya_wave
        self.varyb_wave = varyb_wave
        self.varyzcross_wave = varyzcross_wave
        self.type_wavelet = type_wavelet

        self.a_wave = a_wave_par.value
        self.b_wave = b_wave_par.value
        self.zcross_wave = zcross_wave_par.value
        LCDMCosmology.__init__(self)


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if (self.varya_wave):  l.append(a_wave_par)
        if (self.varyb_wave): l.append(b_wave_par)
        if (self.varyzcross_wave): l.append(zcross_wave_par)
        return l


    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "a_wave":
                self.a_wave = p.value
            elif p.name == "b_wave":
                self.b_wave = p.value
            elif p.name == "zcross_wave":
                self.zcross_wave = p.value
        return True


    #def rhow(self,z):
    #    return self.h*100.0*((self.Om*(1+z)**3+1.0-self.Om)**0.5)

    #def psi0(self,z):
    #    return (-self.a_wave/(2*self.b_wave))*2.73**(-self.b_wave*(z-self.zcross_wave)**2)

    #def psi1(self,z):
    #    return -2*self.b_wave*(z-self.zcross_wave)*self.psi0(z)



    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self,a):
        z= 1./a - 1

        #rhow = self.h*100.0*(((self.Obh2/(self.h**2))*(1+z)**3+self.Om*(1+z)**3+1.0-self.Om-self.Obh2/(self.h**2))**0.5)
        #rhow = self.h*100.0*((self.Om*(1+z)**3+1.0-self.Om)**0.5)
        rhow = self.h*100.0*((self.Ocb/a**3+self.Omrad/a**4+(1.0-self.Om))**0.5)
        
        psi0 = (-self.a_wave/(2*self.b_wave))*np.exp(-self.b_wave*(z-self.zcross_wave)**2) 
        psi1 = -2*self.b_wave*(z-self.zcross_wave)*psi0
        psi2 = 4*self.b_wave*(self.b_wave*(z-self.zcross_wave)**2-0.5)*psi0/10.0
        psi3 = -8*(self.b_wave**2)*(self.b_wave*(z-self.zcross_wave)**3-1.5*(z-self.zcross_wave))*psi0/100.0
        psi4 = 16*(self.b_wave**2)*(0.75+(self.b_wave**2)*(z-self.zcross_wave)**4-3*self.b_wave*(z-self.zcross_wave)**2)*psi0/1000.0

        if self.type_wavelet == 'hermitian_1':
            return ((rhow)/(self.h*100*(1+psi1*rhow)))**2
        elif self.type_wavelet == 'hermitian_2':
            return ((rhow)/(self.h*100*(1+psi2*rhow)))**2
        elif self.type_wavelet == 'hermitian_3':
            return ((rhow)/(self.h*100*(1+psi3*rhow)))**2
        elif self.type_wavelet == 'hermitian_4':
            return ((rhow)/(self.h*100*(1+psi4*rhow)))**2

        #return ((rhow)/(self.h*100*(1+psi1*rhow)))**2

   
