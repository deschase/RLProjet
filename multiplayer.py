import bandit

class Multiplayer(object):
    """ This class is here to model the reaction of the MAB with a draw of chosen arms by some players"""
    def __init__(self, nbPlayers, MAB, model):
        self.M = nbPlayers
        self.MAB = MAB
        self.model = model # this is going to be equal to 1 or 2 : it depends if we choose
        # if the players observe only Y and reward (so also C if Y = 1) or Y and C

    def draw(self, list_arms_chosen):
        rewards = [0. for i in range(self.M)]
        collisions = [False for i in range(self.M)]
        Y = [0 for i in range(self.M)]
        armselected = [0 for i in range(self.MAB.nb_arms)]
        for i in range(self.M):
            armselected[list_arms_chosen[i]] += 1
            Y[i] = self.MAB.list_arms[list_arms_chosen[i]].sample()
        for j in range(self.M):
            if self.model == 1: # observe Y and C
                if armselected[list_arms_chosen[j]] == 1 and Y[j] == 1:
                    rewards[j] = 1.
                elif armselected[list_arms_chosen[j]] > 1:
                    collisions[j] = True
            if self.model == 2: # observe Y and reward
                if armselected[list_arms_chosen[j]] == 1 and Y[j] == 1:
                    rewards[j] = 1.
                if armselected[list_arms_chosen[j]] > 1 and Y[j] == 1: # we know a collision happened only if Y = 1
                    collisions[j] = True

        return rewards, Y, collisions
