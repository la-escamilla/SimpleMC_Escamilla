
import numpy as np
from simplemc.likelihoods.BaseLikelihood import BaseLikelihood
from scipy import constants
import scipy.linalg as la
import scipy as sp


class DR16BAOLikelihood(BaseLikelihood):
    def __init__(self, name, values_filename, fidtheory):
        """
        This module calculates likelihood for the consensus BAODR12.
        BAO-only consensus results, Alam et al. 2016
        https://arxiv.org/abs/1607.03155
        Parameters
        ----------
        name
        values_filename
        cov_filename
        fidtheory

        Returns
        -------

        """
        BaseLikelihood.__init__(self,name)

        self.rd = fidtheory.rd
        print("Loading ", values_filename)
        da = sp.loadtxt(values_filename, usecols = (0,1,2,3))
        self.zs    = da[:, 0]
        self.DM_DH = da[:, 1]
        self.sigma = da[:, 2]
        self.type  = da[:, 3]
        
        print("Loading covariance DR16")
        cov = np.diag(np.square(self.sigma))
        assert(len(cov) == len(self.zs))
        vals, vecs = la.eig(cov)
        vals = sorted(sp.real(vals))
        print("Eigenvalues of cov matrix:", vals[0:3],'...',vals[-1])
        print("Adding marginalising constant")
        cov += 3**2
        self.icov  = la.inv(cov)


    def loglike(self):
        tvec = []
        for i, z in enumerate(self.zs):
            if self.type[i]==4:
                tvec.append(self.theory_.DaOverrd(z))
            elif self.type[i]==5:
                tvec.append(self.theory_.HIOverrd(z))
            elif self.type[i]==3:
                tvec.append(self.theory_.DVOverrd(z))
        tvec = sp.array(tvec)
        #print('hi', self.theory_.DaOverrd(self.zs[0])*self.rd, self.DM_DH[0])
        #print('hi2', constants.c/1000./(self.theory_.HIOverrd(self.zs[1]))/self.rd, self.DM_DH[1])
        #pass
        #tvec = sp.array([100.0*self.theory_.h*sp.sqrt(self.theory_.RHSquared_a(1.0/(1+z))) for z in self.zs])
        #print tvec, self.DM_DH

        ## This is the factor that we need to correct
        ## note that in principle this shouldn't matter too much, we will marginalise over this
        tvec += 0
        delta = tvec - self.DM_DH
        return -sp.dot(delta, sp.dot(self.icov, delta))/2.0


