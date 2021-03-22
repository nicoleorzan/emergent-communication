import Agent
import Config
import Environments
import System

config_dict = {
    "epsilon": 0.01,
    "lr": 0.1
}

config = Config.Config(config_dict)
env = Environments.DiscEnv(num_obs = 3, num_actions = 3)

ag = Agent.DiscAgent(config, env, 0)

num_agents = 2
sys = System.System(num_agents, config, env)
sys.learning_step()