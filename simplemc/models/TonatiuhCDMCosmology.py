import math as N
import numpy as np
from simplemc.models.LCDMCosmology import LCDMCosmology
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from simplemc.cosmo.Parameter import Parameter
from simplemc.cosmo.paramDefs import Ok_par, cmade_par, qcmade_par

class TonatiuhCDMCosmology(LCDMCosmology):
    def __init__(self, varyOk=True, varycmade=True, varyqcmade=True):
        self.varyOk = varyOk
        self.Ok = Ok_par.value
        self.varycmade = varycmade
        self.cmade = cmade_par.value
        self.varyqcmade = varyqcmade
        self.qcmade = qcmade_par.value


        self.zinter = np.linspace(0.0, 3.0, 50)
        self.ninter = np.log(1.0/(1.0+np.linspace(0.0,3.0,50)))
        LCDMCosmology.__init__(self, mnu=0)
        self.updateParams([])


    # my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if (self.varyOk): l.append(Ok_par)
        if (self.varycmade): l.append(cmade_par)
        if (self.varyqcmade): l.append(qcmade_par)
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "cmade":
                self.cmade = p.value
            elif p.name == "qcmade":
                self.qcmade = p.value
            elif p.name == "Ok":
                self.Ok = p.value
                self.setCurvature(self.Ok)
                if (abs(self.Ok) > 1.0):
                    return False

        self.initialize()
        return True

    
    def de_dm_rhow(self, func, n):
        rho_de = func[0]
        rho_dm = func[1]
        #rho_db = func[2]
        #drhodbdz = -3*rho_db
        drhodedz = (self.qcmade)*(1.*np.sqrt(6)/np.pi)*(rho_de**1.5)*(np.exp(-n)/np.sqrt(rho_dm+rho_de+(self.Obh2/(self.h**2))*np.exp(-3*n)+self.Ok*np.exp(-2*n)))
        drhodmdz = -3*rho_dm-self.cmade*drhodedz
        dfuncdz = [drhodedz, drhodmdz]
        return dfuncdz 


    def initialize(self):
        rhowde = np.reshape(odeint(self.de_dm_rhow ,[1.0-self.Om-self.Ok-self.Obh2/(self.h**2), self.Om] ,self.ninter)[:,0],len(self.ninter))
        rhowdm = np.reshape(odeint(self.de_dm_rhow ,[1.0-self.Om-self.Ok-self.Obh2/(self.h**2), self.Om] ,self.ninter)[:,1],len(self.ninter))
        #rhowdb = np.reshape(odeint(self.de_dm_rhow ,[1.0-self.Om-self.Ok-self.Obh2/(self.h**2), self.Om, self.Obh2/(self.h**2)] ,self.ninter)[:,2],len(self.ninter))
        self.rhowde_inter = interp1d(self.ninter, rhowde)
        self.rhowdm_inter = interp1d(self.ninter, rhowdm)
        #self.rhowdb_inter = interp1d(self.ninter, rhowdb)
        return True

    ## this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self,a):
        z= 1./a - 1
        if z>= 3.0:
            om_de = (1.0-self.Om-self.Ok-self.Obh2/(self.h**2))
            om_dm = (self.Om)/a**3
            om_db = (self.Obh2/(self.h**2))/a**3
        else:
            om_de = self.rhowde_inter(np.log(1/(1+z)))
            om_dm = self.rhowdm_inter(np.log(1/(1+z)))
            om_db = (self.Obh2/(self.h**2))/a**3
        if om_dm + self.Omrad/a**4 + om_de + self.Ok/a**2 + om_db is None:
            return 0.1
        elif om_dm+self.Omrad/a**4 + om_de + self.Ok/a**2 + om_db > 0 :
            return om_dm + self.Omrad/a**4 + om_de + self.Ok/a**2 + om_db
        elif om_dm + self.Omrad/a**4 + om_de + self.Ok/a**2 + om_db <=0 :
            return 0.1


