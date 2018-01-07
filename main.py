from bandit import ArmBernoulli, Bandit
from algorithms import MCTopM, MC, RhoRand, MCTopM_with_nbplayer_estim
import numpy as np
import matplotlib.pyplot as plt

random_state = np.random.randint(1, 312414)

arm1 = ArmBernoulli(0.55, random_state=random_state)
arm2 = ArmBernoulli(0.50, random_state=random_state)
arm3 = ArmBernoulli(0.45, random_state=random_state)
arm4 = ArmBernoulli(0.40, random_state=random_state)
arm5 = ArmBernoulli(0.35, random_state=random_state)
arm6 = ArmBernoulli(0.30, random_state=random_state)
arm7 = ArmBernoulli(0.25, random_state=random_state)
arm8 = ArmBernoulli(0.20, random_state=random_state)
arm9 = ArmBernoulli(0.15, random_state=random_state)
arm10 = ArmBernoulli(0.10, random_state=random_state)

arms = Bandit([arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, arm9, arm10])
nb_player = 4
model = 2

nbExperiment = 50
rwd1 = []
rwd2 =[]
rwd3 = []
rwd4 = []
for i in range(nbExperiment):
    game1 = MCTopM(nb_player, model, arms)
    game1.launch_game(5010)
    rwd1.append(game1.regret)

    # note : this one has no idea of the nb of players
    game2 = MC(nbPlayers = nb_player, model = 1, MAB = arms,T0 = 1000, T1 = 4010)
    game2.launch_game()
    rwd2.append(game2.regret)

    game3 = RhoRand(nbPlayers = nb_player, model = 1, MAB = arms)
    game3.launch_game(5000)
    rwd3.append(game3.regret)

    # This one either
    game4 = MCTopM_with_nbplayer_estim(nb_player, model, arms, 5.)
    game4.launch_game(5000)
    rwd4.append(game4.regret)



rgt1 = np.asarray(rwd1)
rgt1 = np.mean(rgt1, axis = 0)
rgt2 = np.asarray(rwd2)
rgt2 = np.mean(rgt2, axis = 0)
rgt3 = np.asarray(rwd3)
rgt3 = np.mean(rgt3, axis = 0)
rgt4 = np.asarray(rwd4)
rgt4 = np.mean(rgt4, axis = 0)
plt.plot(np.cumsum(rgt1), c= 'g')
plt.plot(np.cumsum(rgt2), c= 'b')
plt.plot(np.cumsum(rgt3), c= 'r')
plt.plot(np.cumsum(rgt4), c= 'orange')
plt.show()