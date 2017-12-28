from multiplayer import Multiplayer
import bandit
import random as rd
import numpy as np
import math

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
        """ Choose the next arms to select for each player"""
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
        """
        We calculate the bound value for ucb.
        :param j_play: the nb of the player that we calculate the estimated bounds of
        :return: computes the bounds in estim2 and keep the old ones in estim1
        """
        for k in range(self.multiplayer.MAB.nb_arms):
            self.estim1[j_play,k] = self.estim2[j_play,k]
            if self.T[j_play,k] != 0:
                mu_k = self.S[j_play,k]/float(self.T[j_play,k])
                self.estim2[j_play,k] = mu_k + np.sqrt((np.log(self.t)/(2*self.T[j_play,k]))) # for the moment we only coded with UCB1 : to test with kl-UCB
            else:
                self.estim2[j_play,k] = 1000000

    def compute_estim(self):
        """ Compute the bounds for each player """
        for j in range(self.nbPlayers):
            self.logUCB(j)

    def compute_M(self):
        """ Computes the M best arms for each player """
        for j in range(self.nbPlayers):
            for a in range(self.nbPlayers):
                self.M[j] = np.argsort(self.estim2[j,:])[::-1][:self.nbPlayers]

    def play_arms(self):
        """ Draw the selected arms for each player and get he info from this draw"""
        rew, Y, self.C = self.multiplayer.draw(self.A)
        for j in range(self.nbPlayers):
            self.S[j, self.A[j]] += Y[j]
            self.T[j, self.A[j]] += 1

    def lauch_game(self, horizon):
        """ Launch the algorithm on a time horizon and return the last arms chosen """
        while self.t < horizon:
            self.choose_arms()
            self.t += 1
            self.play_arms()
            self.compute_estim()
            self.compute_M()
        return self.A

class MC(object):
    """ This class is the exact algorithm of musical chair from the article
    "Multi-player Bandits : A Musical Chairs Approach", ie no knowledge of the nb of players during the algo """
    def __init__(self, nbPlayers, model, MAB,T0, T1):
        self.multiplayer = Multiplayer(nbPlayers, MAB, model)
        self.nbPlayers = nbPlayers # To be able to compare with the estimation of nb of players we are going to make
        self.T0 = T0 # Time for initialisation
        self.T1 = T1 # Total horizon of the experiment
        self.K = MAB.nb_arms  # the total number of arms
        self.N = [0 for i in range(self.nbPlayers)]  # estimated nb of players

        self.C_T0 = [0 for i in range(self.nbPlayers)] # Number of collision during the time T0 for each player
        self.o = np.zeros([nbPlayers, self.K]) # nb of time an arm is drawn without collision
        self.mu = np.zeros([nbPlayers, self.K]) # estimated reward for each arm
        self.s = np.zeros([nbPlayers, self.K]) # cumulated reward
        self.A = np.zeros([nbPlayers, self.K]) # is going to contain the arms ranked after T0

        self.fixed = [False for i in range(self.nbPlayers)] # To know if you can change the arm chosen or not
        self.Chosen = [rd.randint(0, self.K - 1) for i in range(nbPlayers)] # the arms chosen by the players
        self.C = [False for i in range(nbPlayers)] # Collisions
        self.t = 0 # just to know the iterations nb

        self.regret = list() # to keep the total regret at each step
        self.max = np.sum(-np.sort(-np.asarray(MAB.means))[0:self.nbPlayers]) # mean of the max result : nbPlayers best arms chosen

    def initialisation(self):
        """ Time for estimation of the mu and the nb of players """
        while self.t < self.T0:
            self.t += 1
            self.Chosen = [rd.randint(0, self.K - 1) for i in range(self.nbPlayers)]
            rew, Y, self.C = self.multiplayer.draw(self.Chosen)
            for j in range(self.nbPlayers):
                if self.C[j]:
                    self.C_T0[j] += 1
                else:
                    self.o[j, self.Chosen[j]] += 1
                    self.s[j, self.Chosen[j]] += rew[j]
            self.regret.append(self.max - sum(rew))

        # After the loop : estimation of the nb of players and the mu
        self.mu = np.asarray([[float(s)/max(1.,float(o)) for s,o in zip(self.s[j], self.o[j])] for j in range(self.nbPlayers)])
        self.A = np.argsort(-self.mu) # sort the mus
        for j in range(self.nbPlayers):
            if self.C_T0[j] == self.T0:
                self.N[j] = self.K
            else:

                self.N[j] = round(math.log(float(self.T0 - self.C_T0[j])/float(self.T0)) / math.log(1. - 1./float(self.K)) + 1.)

    def musical_chair(self):
        """ Second moment of the algo : after estimating the ranks of the "chairs", the players
        randomly choose one in the N[j]  (nb of player estimated) best and keep it if the first
        time she chose it there was no collision """
        while self.t < self.T1:
            self.t += 1
            for j in range(self.nbPlayers):
                if not self.fixed[j]:
                    self.Chosen[j] = np.random.choice(self.A[j, 0:int(self.N[j])])
            rew, Y, self.C = self.multiplayer.draw(self.Chosen)
            self.regret.append(self.max - sum(rew))
            for j in range(self.nbPlayers):
                if not self.C[j]:
                    self.fixed[j] = True

    def launch_game(self):
        self.initialisation()
        self.musical_chair()

class RhoRand(object):
    """ This class is exactly the algo described in "Distributed Algorithm for Learning and Cognitive Medium
    Access with Logarithmic Regret" in the part with a known number of player """
    def __init__(self, nbPlayers, model, MAB):
        self.multiplayer = Multiplayer(nbPlayers, MAB, model)
        self.nbPlayers = nbPlayers # To be able to compare with the estimation of nb of players we are going to make
        self.C = MAB.nb_arms  # the total number of arms
        self.channel_initialised = [[] for j in range(self.nbPlayers)]


        self.T = np.zeros([nbPlayers, self.C]) # nb of time an arm is drawn
        self.s = np.zeros([nbPlayers, self.C])  # cumulated reward for each player and each arm
        self.X = np.zeros([nbPlayers, self.C]) # estimated reward for each arm
        self.g = np.zeros([nbPlayers, self.C])

        self.A = np.zeros([nbPlayers, self.C]) # is going to contain the arms ranked
        self.Cur_rank = [1 for i in range(nbPlayers)] # Rank chosen by the player
        self.Curr_selected = [0 for i in range(nbPlayers)] # Arm chosen dur to the rank

        self.collision = [False for i in range(nbPlayers)] # Collisions
        self.t = 0 # just to know the iterations nb

        self.regret = list() # to keep the total regret at each step
        self.max = np.sum(-np.sort(-np.asarray(MAB.means))[0:self.nbPlayers]) # mean of the max result : nbPlayers best arms chosen

    def initialisation(self):
        while self.t < self.C:
            self.t += 1
            for j in range(self.nbPlayers):
                # We explore every channel
                i = rd.randint(0, self.C - 1)
                while i in self.channel_initialised[j]:
                    i = rd.randint(0, self.C - 1)
                self.Curr_selected[j] = i
                self.channel_initialised[j].append(i)
            print self.Curr_selected
            rew, Y, self.collision = self.multiplayer.draw(self.Curr_selected)
            self.regret.append(self.max - sum(rew)) # keep the regret in mind for all the process
            for j in range(self.nbPlayers):
                self.s[j, self.Curr_selected[j]] += rew[j]
                self.T[j, self.Curr_selected[j]] += 1.

        # We estimate the  Xj
        self.X = np.asarray([[float(s)/float(t) for s, t in zip(self.s[j], self.T[j])] for j in range(self.nbPlayers)])
        self.collision = [False for i in range(self.nbPlayers)]

    def bound_estimate(self):
        for j in range(self.nbPlayers):
            for i in range(self.C):
                self.g[j, i] = self.X[j, i] + min(math.sqrt(math.log(float(self.t))/float(2*self.T[j, i])), 1)

    def rhoRhand(self, horizon):
        self.bound_estimate()
        self.A = np.argsort(-self.g) # we rank the arms
        self.Curr_selected = [int(self.A[j, int(k)]) for k, j in zip(self.Cur_rank, range(self.nbPlayers))]
        while self.t < horizon:
            self.t += 1
            rew, Y, self.collision = self.multiplayer.draw(self.Curr_selected)
            self.regret.append(self.max - sum(rew))  # keep the regret in mind for all the process
            for j in range(self.nbPlayers):
                self.s[j, self.Curr_selected[j]] += rew[j]
                self.T[j, self.Curr_selected[j]] += 1.

                if self.collision[j]: # if there was a collision, we change the rank chosen
                    self.Cur_rank[j] = rd.randint(0, self.nbPlayers - 1)
            # We estimate the  Xj
            self.X = np.asarray([[s / t for s, t in zip(self.s[j], self.T[j])] for j in range(self.nbPlayers)])
            self.bound_estimate()
            self.A = np.argsort(-self.g)  # we rank the arms
            self.Curr_selected = [int(self.A[j, int(k)]) for k, j in zip(self.Cur_rank, range(self.nbPlayers))]# next arms to be selected

    def launch_game(self, horizon):
        self.initialisation()
        self.rhoRhand(horizon)











