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
        self.M = np.zeros([nbPlayers, nbPlayers], dtype=np.int) # the M best arms at each row
        # all the things to compute self.M :
        self.T = np.zeros([nbPlayers, MAB.nb_arms])
        self.S = np.zeros([nbPlayers, MAB.nb_arms])
        self.estim1 = np.zeros([nbPlayers, MAB.nb_arms])
        self.estim2 = np.zeros([nbPlayers, MAB.nb_arms])

    def choose_arms(self):
        # for self.t == 0, we just take the A randomly (in the initialization)
        if self.t != 0:
            for j in range(self.nbPlayers):
                if not (self.A[j] in self.M[j,:]):
                    set_possib = [k for k in self.M[j] if (self.estim1[j,k] <= self.estim1[j, self.A[j]])]
                    self.A[j] = rd.choice(set_possib)
                    self.s[j] = False
                elif self.C[j] and not self.s[j]:
                    self.A[j] = rd.choice(self.M[j])
                    self.s[j] = False
                else:
                    self.A[j] = self.A[j]
                    self.s[j] = True

    def logUCB(self, j_play):
        # We calculate the bound value for ucb.
        for k in range(self.multiplayer.MAB.nb_arms):
            self.estim1[j_play,k] = self.estim2[j_play,k]
            if self.T[j_play,k] != 0:
                mu_k = self.S[j_play,k]/float(self.T[j_play,k])
                self.estim2[j_play,k] = mu_k + np.sqrt((np.log(self.t)/(2*self.T[j_play,k])))
            else:
                self.estim2[j_play,k] = 1000000

    def compute_estim(self):
        for j in range(self.nbPlayers):
            self.logUCB(j)

    def compute_M(self):
        for j in range(self.nbPlayers):
            for a in range(self.nbPlayers):
                self.M[j] = np.argsort(self.estim2[j,:])[::-1][:self.nbPlayers]

    def play_arms(self):
        rew, Y, self.C = self.multiplayer.draw(self.A)
        for j in range(self.nbPlayers):
            self.S[j, self.A[j]] += Y[j]
            self.T[j, self.A[j]] += 1

    def lauch_game(self, horizon):
        while self.t < horizon:
            self.choose_arms()
            self.t += 1
            self.play_arms()
            self.compute_estim()
            self.compute_M()
        return self.A
