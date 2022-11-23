import gym
import a3_gym_env
import Modules
import torch_misc
env = gym.make('Pendulum-v1-custom')

# sample hyperparameters
batch_size = 10000
epochs = 30
learning_rate = 1e-2
hidden_size = 8
n_layers = 2


#Step 1: Start by implementing an environment interaction loop. You may refer to homework 1 for inspiration. 
def EnvironmentIterationLoop(iterations):
    env=gym.make("Pendulum-v1-custom")
    obs = env.reset()

    for i in range(iterations):
        obs, reward, done, info = env.step([0])
        print(obs)
    return 0

print(EnvironmentIterationLoop(5))