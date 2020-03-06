from movielens import MovieLens
from linUCB import LinUCB
import numpy as np
import matplotlib.pyplot as plt

ucb = LinUCB(alpha=0.1, dataset=None, max_items=5, allow_selecting_known_arms=True)
avg_reward = ucb.run(num_epochs=2)


