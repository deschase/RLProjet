from bandit import ArmBernoulli, Bandit

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
