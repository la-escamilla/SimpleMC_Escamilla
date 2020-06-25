"""BAMBI management object.

Author: Pat Scott (p.scott@imperial.ac.uk)
Date: Feb 2019

Modified for SimpleMC use by I Gomez-Vargas (igomezv0701@alumno.ipn.mx)
Date: June 2020
"""

import numpy as np
from simplemc.analyzers.pybambi.kerasnet import KerasNetInterpolation
# import keras.models
try:
    import tensorflow as tf
except:
    sys.exit("You need to install tensorflow")


class BambiManager(object):
    """Does all the talking for BAMBI.

    Takes a new set of training data from the dumper and trains (or retrains) a
    neural net, and assesses whether or not it can be used for a given
    parameter combination.

    Parameters
    ----------
    ntrain: int
        Number of training points to use

    """

    def __init__(self, loglikelihood, learner, proxy_tolerance,
                 failure_tolerance, ntrain, split=0.8, numNeurons=200, epochs=300,
                 model=None, savedmodelpath=None, it_to_start_net=1000, updInt=500):
        """Construct bambi object."""

        self.proxy_tolerance = proxy_tolerance
        self._loglikelihood = loglikelihood
        self._learner = learner
        self._proxy_tolerance = proxy_tolerance
        self._failure_tolerance = failure_tolerance
        self._ntrain = ntrain
        self._proxy_trained = False
        self.old_learners = []
        self.split = split
        self.numNeurons = numNeurons
        self.epochs = epochs
        self.model = model
        self.savedmodelpath = savedmodelpath
        self.dumpercount = 0
        self.it_to_start_net = it_to_start_net
        self.updInt = updInt

    def make_learner(self, params, loglikes):
        """Construct a Predictor."""
        if self._learner == 'keras':
            return KerasNetInterpolation(params, loglikes,
                                         split=self.split, numNeurons=self.numNeurons,
                                         epochs=self.epochs, model=self.model,
                                         savedmodelpath=self.savedmodelpath)
        elif self._learner == 'nearestneighbour':
            return NearestNeighbourInterpolation(params, loglikes)
        elif issubclass(type(self._learner), tf.keras.models.Model):
            return KerasNetInterpolation(params, loglikes, model=self._learner)
        else:
            raise NotImplementedError('learner %s is not implemented.'
                                      % self._learner)

    def dumper(self, live_params, live_loglks, dead_params, dead_loglks):
        self.dumpercount += 1
        # print("Respond to signal from nested sampler.")
        if not self._proxy_trained and self.dumpercount >= self.it_to_start_net:
            mod_it_after_net = (self.dumpercount - self.it_to_start_net)%self.updInt
            if self.dumpercount == self.it_to_start_net or mod_it_after_net == 0:
                dead_params = np.array(dead_params)
                dead_loglks = np.array(dead_loglks)
                params = np.concatenate((live_params, dead_params))
                loglikes = np.concatenate((live_loglks, dead_loglks))
                self.train_new_learner(params[:self._ntrain, :],
                                        loglikes[:self._ntrain])

        if self._proxy_trained:
            print("\rUsing trained proxy ")
        else:
            print("\rUnable to use proxy ")

    def loglikelihood(self, params):
        """Bambi Proxy wrapper for original loglikelihood."""
        # Short circuit to the full likelihood if proxy not yet fully trained
        if not self._proxy_trained:
            return self._loglikelihood(params)

        # Call the learner
        candidate_loglikelihood = self._current_learner(params)

        # If the learner can be trusted, use its estimate,
        # otherwise use the original like and update the failure status
        if self._current_learner.valid(candidate_loglikelihood):
            return candidate_loglikelihood
        else:
            self._rolling_failure_fraction = (1.0 + (self._ntrain - 1.0) *
                                              self._rolling_failure_fraction
                                              ) / self._ntrain
            if self._rolling_failure_fraction > self._failure_tolerance:
                self._proxy_trained = False
            return self._loglikelihood(params)

    def train_new_learner(self, params, loglikes):
        """Train a new Predictor."""
        try:
            self.old_learners.append(self._current_learner)
        except AttributeError:
            pass
        self._current_learner = self.make_learner(params, loglikes)
        sigma = self._current_learner.uncertainty()
        print("Current uncertainty in network log-likelihood predictions: %s"
              % sigma)
        if sigma < self._proxy_tolerance:
            self._proxy_trained = True
            self._rolling_failure_fraction = 0.0