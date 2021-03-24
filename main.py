from agents.Agent import DiscAgent
from utils.Config import Config
from utils.Environments import DiscEnv
import System

config_dict = {
    "epsilon": 0.01,
    "lr": 0.1
}

config = Config(config_dict)
env = DiscEnv(num_obs = 3, num_actions = 3)

ag = DiscAgent(config, env, 0)

num_agents = 2
sys = System.System(num_agents, config, env)
#sys.interaction_step()
sys.learning_loop(50000)

ags = sys.return_dict_agents()

sys.test_loop(1000)