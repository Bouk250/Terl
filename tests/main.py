import time

from tests.env import mack_vec_env
from tests.custom_indicators import *
from stable_baselines3.common.evaluation import evaluate_policy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch

def main_func():
    env = mack_vec_env("TestEnvSingle", 1, "cuda")
    obs = env.reset()
    obs = np.expand_dims(obs, axis=-1)
    print(obs.shape)
    """
    fig = make_subplots(1, obs.shape[0])
    for i in range(obs.shape[0]):
        fig.add_trace(px.imshow(obs[i]).data[0], 1, i+1)
    #fig = px.imshow(obs[0,0])
    fig.show()
    """

if __name__ == '__main__':
    main_func()
