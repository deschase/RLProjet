from bandit import ArmBernoulli, Bandit
from algorithms import MCTopM, MC, RhoRand
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
model = 1

#game1 = MCTopM(nb_player, model, arms)
#print game1.lauch_game(10000)

game2 = MC(nbPlayers = nb_player, model = 1, MAB = arms,T0 = 10, T1 = 100)
game2.launch_game()
plt.plot(np.cumsum(game2.regret), c= 'b')

game3 = RhoRand(nbPlayers = nb_player, model = 1, MAB = arms)
game3.launch_game(100)
plt.plot(np.cumsum(game3.regret), c= 'r')

plt.show()