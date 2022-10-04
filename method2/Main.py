import pickle

import pandas as pd

from JacobianMatrix import run
import numpy as np

data = run("heartbeat", "LR", 0.01, 20000, seed=14, label=0)

df = pd.DataFrame(data)

df.to_csv("/Users/uma.kannikanti/PycharmProjects/membershipinferenceattacks/method2/data/attack/train_0.csv", header=False, index=False)



