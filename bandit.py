import numpy as np

class AbstractArm(object):
    def __init__(self, mean, variance, random_state):
        """
        Args:
            mean: expectation of the arm
            variance: variance of the arm
            random_state (int): seed to make experiments reproducible
        """
        self.mean = mean
        self.variance = variance

        self.local_random = np.random.RandomState(random_state)

    def sample(self):
        pass

class ArmBernoulli(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        Bernoulli arm
        Args:
             p (float): mean parameter
             random_state (int): seed to make experiments reproducible
        """
        self.p = p
        super(ArmBernoulli, self).__init__(mean=p,
                                           variance=p * (1. - p),
                                           random_state=random_state)

    def sample(self):
        return self.local_random.rand(1) < self.p

class Bandit(object):
    def __init__(self, list_arms):
        self.list_arms = list_arms
        self.nb_arms = len(MAB)
        self.means = [el.mean for el in MAB]
