## This is phiCDM cosmology

import numpy as np
from LCDMCosmology import *
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from ParamDefs import mphi_par


class PhiCDMCosmology(LCDMCosmology):
    def __init__(self, varymphi = True):
        ## two parameters: Om and h

        """Is better to start the chains at masses equal one, othewise
        may take much longer"""

        self.varymphi  = varymphi

        LCDMCosmology.__init__(self, mnu=0)
        self.lna   = np.linspace(-13, 0, 200)
        self.z     = np.exp(-self.lna) - 1.
        self.zvals = np.linspace(0, 3, 100)

        self.mphi  = 1.94
        self.alpha = 0.2
        self.cte   = 3.0*self.h**2

        self.updateParams([])

    ## my free parameters. We add Ok on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l=LCDMCosmology.freeParameters(self)
        if (self.varymphi)  : l.append(mphi_par)
        return l


    def updateParams(self,pars):
        ok=LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name  == "mphi":
                self.mphi= p.value

        self.search_ini()

        """
        dataHz = np.loadtxt('data/Hz_all.dat')
        redshifts, obs, errors = [dataHz[:,i] for i in [0,1,2]]
        plt.errorbar(redshifts, obs, errors, xerr=None,
                     color='purple', marker='o', ls='None',
                     elinewidth =2, capsize=5, capthick = 1, label='$Datos$')
        plt.xlabel(r'$z$')
        plt.ylabel(r'$H(z) [km/s Mpc^{-1}]$')
        plt.plot(self.zvals, 100*self.hub_SF_z)
        plt.title('mquin %f'%(self.mphi))
        plt.show()
        """
        return True


    def Vtotal(self, x, select):
        """Cuadratic potential and its derivatives wrt phi or psi"""
        if select == 0:
            Vtotal = 0.5*(x*self.mphi)**2
        elif select == 'phi':
            Vtotal = x*self.mphi**2
        return Vtotal


    def rhode(self, x_vec):
        quin, dotquin= x_vec
        Ode = 0.5*dotquin**2 + self.Vtotal(quin, 0) / self.cte
        return Ode


    def eos(self, x_vec):
        quin, dotquin = x_vec
        w1 = 0.5*dotquin**2  - self.Vtotal(quin, 0)/self.cte
        w2 = 0.5*dotquin**2  + self.Vtotal(quin, 0)/self.cte
        return w1/w2



    def hubble(self, lna, x_vec=None, SF = True):
        a = np.exp(lna)
        if SF:
            Ode = self.rhode(x_vec)
        else:
            Ode = 1.0-self.Om
        return self.h*np.sqrt(self.Ocb/a**3 + self.Omrad/a**4 + Ode)


    def logatoz(self, func):
        # change function from lna to z
        tmp     = interp1d(self.lna, func)
        functmp = tmp(self.lna)
        final   = np.interp(self.zvals, self.z[::-1], functmp[::-1])
        return final



    def RHS(self, x_vec, lna):
        sqrt3H0 = np.sqrt(self.cte)
        quin, dotquin= x_vec
        hubble  = self.hubble(lna, x_vec)
        return [sqrt3H0*dotquin/hubble, -3*dotquin - self.Vtotal(quin, 'phi')/(sqrt3H0*hubble)]



    def solver(self, quin0, dotquin0):
        y0       = [quin0, dotquin0]
        y_result = odeint(self.RHS, y0, self.lna, h0=1E-10)
        return y_result



    def calc_Ode(self, mid):
        """Select Quintess, Phantom or Quintom"""
        sol = self.solver(mid, 0.0).T
        quin, dotq = sol
        rho = self.rhode(sol)[-1]

        Ode = rho*(self.h/self.hubble(0.0, [quin[-1], dotq[-1]]))**2
        tol = (1- self.Ocb- self.Omrad) - Ode
        return sol, tol, Ode


    def bisection(self):
        """Search for intial condition of phi such that \O_DE today is 0.7"""
        lowphi, highphi = -10, 10 #200
        Ttol            = 1E-5 #2
        mid = (lowphi + highphi)/2.0
        while (highphi - lowphi )/2.0 > Ttol:
            sol, tol_mid, Ode = self.calc_Ode(mid)
            tol_low = self.calc_Ode(lowphi)[1]
            if(np.abs(tol_mid) < Ttol):
                  print ('reach tolerance',  'phi_0=', mid, 'error=', tol_mid)
                  return mid, sol
            elif tol_low*tol_mid<0:
                highphi  = mid
            else:
                lowphi   = mid
            mid = (lowphi + highphi)/2.0

        #print 'No solution found!', mid, self.mquin, self.mphan
        ##Check whether rho is constant or nearby
        grad = np.abs(np.gradient(self.rhode(sol))).max()
        self.mid = mid
        mid  = -1 if grad < 1.0E-2 else 0
        return mid, sol


    def search_ini(self):
        mid, sol = self.bisection()
        if (mid == 0):
            #print ('mass=',  self.mphi, 'solution not found with', self.mid)
            self.hub_SF   = interp1d(self.lna, self.hubble(self.lna, SF=False))
            #self.w_eos    = interp1d(self.lna, -1*np.ones(len(self.lna)))
            #self.hub_SF_z = self.logatoz(self.lna, np.ones(len(self.lna)))
        elif (mid == -1):
            #print ('looks like LCDM')
            self.hub_SF   = interp1d(self.lna, self.hubble(self.lna, SF=False))
            #self.w_eos    = interp1d(self.lna, -1*np.ones(len(self.lna)))
            #self.hub_SF_z = self.logatoz(self.hubble(self.lna, sol, SF=False))
        else:
            self.hub_SF   = interp1d(self.lna, self.hubble(self.lna, sol))
            #self.w_eos    = interp1d(self.lna, self.eos(sol))
            #self.hub_SF_z = self.logatoz(self.hubble(self.lna, sol))
        return True



    ## this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self,a):
        lna = np.log(a)
        if (1./a-1 < 10):
            hubble = (self.hub_SF(lna)/self.h)**2.
        else:
            hubble = self.hubble(self.lna, SF=False)
        return hubble


    def Hubble_a(self, a):
        lna = np.log(a)
        hubble = 100*self.hub_SF(lna)
        return hubble


    def w_de(self, a):
        lna = np.log(a)
        return self.w_eos(lna)


    #def RHSquared_z(self, z):
    #    hubble = (self.hub_SF_z(z)/self.h)**2.
    #    return hubble
