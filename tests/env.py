from terl.envs import TradingEnv
from terl.db import DBManager
from tests.wrappers import ForexWrapper, VecPyTorch
from stable_baselines3.common.env_util import DummyVecEnv, SubprocVecEnv
from terl.config import EnvConfigManager

def make_env(env_id, rank, seed, dbm=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = ForexWrapper(TradingEnv(env_id, dbm))
        env.seed(seed + rank)
        return env
    return _init

def mack_vec_env(env_id, num_env, dvice, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    """

    # Number of processes to use
    # Create the vectorized environment
    conf = EnvConfigManager().get_config(env_id)
    dbm = DBManager(conf.get('db'))

    venvs = SubprocVecEnv([make_env(env_id, i, seed, dbm) for i in range(num_env)])
    return venvs