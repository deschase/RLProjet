from bandit import ArmBernoulli, Bandit
from algorithms import MCTopM, MC, RhoRand, MCTopM_with_nbplayer_estim, MCTopM_with_time_estimation
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

"""arm1 = ArmBernoulli(0.95, random_state=random_state)
arm2 = ArmBernoulli(0.9, random_state=random_state)
arm3 = ArmBernoulli(0.8, random_state=random_state)
arm4 = ArmBernoulli(0.7, random_state=random_state)
arm5 = ArmBernoulli(0.6, random_state=random_state)
arm6 = ArmBernoulli(0.5, random_state=random_state)
arm7 = ArmBernoulli(0.4, random_state=random_state)
arm8 = ArmBernoulli(0.3, random_state=random_state)
arm9 = ArmBernoulli(0.2, random_state=random_state)
arm10 = ArmBernoulli(0.1, random_state=random_state)"""

arms = Bandit([arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, arm9, arm10])
nb_player = 4
model = 1

nbExperiment = 50
rwd1 = []
rwd1bis = []
rwd2 =[]
rwd3 = []
rwd3bis = []
rwd4 = []
rwd4bis = []
rwd5 = []
horizon = 8000
T0 = 4000
T0_bis = 1500
for i in range(nbExperiment):
    """game1 = MCTopM(nb_player, model, arms)
    game1.launch_game(horizon)
    rwd1.append(game1.regret)"""

    """game1bis = MCTopM(nb_player, model, arms, UCB=False)
    game1bis.launch_game(horizon)
    rwd1bis.append(game1bis.regret)"""

    # note : this one has no idea of the nb of players
    """game2 = MC(nbPlayers = nb_player, model = 1, MAB = arms,T0 = T0, T1 = horizon)
    game2.launch_game()
    rwd2.append(game2.regret)"""

    """game3 = RhoRand(nbPlayers = nb_player, model = model, MAB = arms)
    game3.launch_game(horizon)
    rwd3.append(game3.regret)"""

    """game3bis = RhoRand(nbPlayers=nb_player, model=model, MAB=arms, UCB=False)
    game3bis.launch_game(horizon)
    rwd3bis.append(game3bis.regret)"""

    # This one either
    """game4 = MCTopM_with_nbplayer_estim(nb_player, model, arms, update=True)
    game4.launch_game(horizon)
    rwd4.append(game4.regret)"""

    game4bis = MCTopM_with_nbplayer_estim(nb_player, model, arms)
    game4bis.launch_game(horizon)
    rwd4bis.append(game4bis.regret)

    # This one either
    game5 = MCTopM_with_time_estimation(nb_player, model, arms)
    game5.launch_game(horizon, T0_bis)
    rwd5.append(game5.regret)



"""rgt1 = np.asarray(rwd1)
rgt1 = np.mean(rgt1, axis = 0)
rgt1bis = np.asarray(rwd1bis)
rgt1bis = np.mean(rgt1bis, axis = 0)"""
"""rgt2 = np.asarray(rwd2)
rgt2 = np.mean(rgt2, axis = 0)"""
"""rgt3 = np.asarray(rwd3)
rgt3 = np.mean(rgt3, axis = 0)
rgt3bis = np.asarray(rwd3bis)
rgt3bis = np.mean(rgt3bis, axis = 0)"""
"""rgt4 = np.asarray(rwd4)
rgt4 = np.mean(rgt4, axis = 0)"""
rgt4bis = np.asarray(rwd4bis)
rgt4bis = np.mean(rgt4bis, axis = 0)
rgt5 = np.asarray(rwd5)
rgt5 = np.mean(rgt5, axis = 0)
#plt.plot(np.cumsum(rgt1), c= 'g', label = "MCTopM-UCB")
#plt.plot(np.cumsum(rgt1bis), c= 'purple', label = "MCTopM-klUCB")
#plt.plot(np.cumsum(rgt2), c= 'b', label = "Musical Chair Algorithm")
#plt.plot(np.cumsum(rgt3), c= 'r', label = "Rhorand-UCB")
#plt.plot(np.cumsum(rgt3bis), c = 'pink', label = "Rhorand-klUCB")
#plt.plot(np.cumsum(rgt4), c= 'orange', label = "MCTop with dynamic estimation")
plt.plot(np.cumsum(rgt4bis), c= 'g', label = "MCTop with dynamic estimation")
plt.plot(np.cumsum(rgt5), c= 'r', label = "MCTopM with evaluation time")
plt.title("Evolution of the cumulated regret (mean on 50 simulations) for the model 1 and 4 players")
plt.legend()
plt.show()