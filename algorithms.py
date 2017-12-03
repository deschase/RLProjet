from multiplayer import Multiplayer
import random as rd
import numpy as np

class MCTopM(object):
    def __init__(self, nbPlayers, model, MAB):
        self.multiplayer = Multiplayer(nbPlayers, MAB, model)
        self.nbPlayers = nbPlayers
        self.s = [False for i in range(nbPlayers)]
        self.A = [rd.randint(0, MAB.nb_arms - 1) for i in range(nbPlayers)]
        self.C = [False for i in range(nbPlayers)]
        self.t = 0 # just to know the iterations nb
        self.M = np.zeros([ nbPlayers, nbPlayers]) # the M best arms at each row
        # all the things to compute self.M :
        self.indices = np.zeros([nbPlayers, MAB.nb_arms])
        self.T = np.zeros([nbPlayers, MAB.nb_arms])
        self.S = np.zeros([nbPlayers, MAB.nb_arms])

    def choose_arms(self):
        # for self.t == 0, we just take the A randomly (in the initialization)
        if self.t != 0:
            for j in range(self.nbPlayers):
                if not (self.A[j] in self.M[j,:]):
                    pass # to finish ->follow algo
                    #self.A[j] = rd.choice()

