

from simplemc.models.LCDMCosmology import LCDMCosmology
from simplemc.cosmo.paramDefs import zt_par


class LsCDMCosmology(LCDMCosmology):
    def __init__(self, varyzt=True):
        """
        This is a CDM cosmology with constant eos w for DE
        Parameters
        ----------
        varyw

        Returns
        -------

        """

        self.varyzt = varyzt
        self.zt = zt_par.value
        LCDMCosmology.__init__(self)


    # my free parameters. We add w on top of LCDM ones (we inherit LCDM)
    def freeParameters(self):
        l = LCDMCosmology.freeParameters(self)
        if (self.varyzt): l.append(zt_par)
        return l

    def updateParams(self, pars):
        ok = LCDMCosmology.updateParams(self, pars)
        if not ok:
            return False
        for p in pars:
            if p.name == "zt":
                self.zt = p.value
        return True


    # this is relative hsquared as a function of a
    ## i.e. H(z)^2/H(z=0)^2
    def RHSquared_a(self, a):
        z= 1./a - 1
        NuContrib = self.NuDensity.rho(a)/self.h**2
        if z<=self.zt:
            return (self.Ocb/a**3+self.Omrad/a**4+NuContrib+(1.0-self.Om))
        elif z>self.zt:
            return (self.Ocb/a**3+self.Omrad/a**4+NuContrib+(1.0-self.Om)*(-1.0))

