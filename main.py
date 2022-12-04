import gym
import a3_gym_env
import Modules
import torch_misc
import sys

env = gym.make('Pendulum-v1-custom')
import time
import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
from Modules import NormalModule
from Modules import PendulumNN
from Modules import PastPendulumNN
from torch_misc import vectorized_vs_nonvectorized
import numpy
import pandas as pd
# sample hyperparameters
batch_size = 10000
epochs = 30
learning_rate = 1e-2
hidden_size = 8
n_layers = 2
clipEpsilon = 0.2

class PPOModels():
    def __init__(self):
        self.Actor = PendulumNN(3, 1)
        self.Critic = PendulumNN(3,1)
        self.optimActor = Adam(self.Actor.parameters(), lr=3e-4)
        self.optimCritic = Adam(self.Critic.parameters(), lr=3e-4)

    def saveModels(self):
        torch.save(self.Actor.state_dict(),"ActorModel.pt")
        torch.save(self.Critic.state_dict(),"CriticModel.pt")
        torch.save(self.optimActor.state_dict(),"optimActor.pt")
        torch.save(self.optimCritic.state_dict(),"optimCritic.pt")


class VanillaModel():
    def __init__(self):
        self.Actor = PendulumNN(3, 1)
        self.optimActor = Adam(self.Actor.parameters(), lr=0.1)

    def saveModels(self):
        torch.save(self.Actor.state_dict(),"VanillaModel.pt")
        torch.save(self.optimActor.state_dict(),"optimVanilla.pt")
        
def RewardsToGo(rewards,discountFactor):
    if len(rewards) == 1:
        return rewards
    RewardNext = RewardsToGo(rewards[1:],discountFactor)
    return [rewards[0]+(1-discountFactor)*RewardNext[0]]+RewardNext


def DoRollout(Models,rollout_size,episode_size,discountFactor):
    obervations =[]
    rewards = []
    actions = []
    logprobs = []
    for i in range(rollout_size):
        episodeRewards = []
        obs = env.reset()
        for j in range(episode_size):
            out_mean,out_variance   = Models.Actor(torch.as_tensor(obs))
            out_action_distribution = Normal(out_mean, out_variance)
            action                  = out_action_distribution.sample()
            obervations.append(obs)
            obs, reward, done, info = env.step(action.detach())
            logprob = out_action_distribution.log_prob(action).detach()
            episodeRewards.append(reward.tolist())
            actions.append(action.numpy())
            logprobs.append(logprob)
        rewards.append(episodeRewards)
    return torch.tensor(numpy.array(obervations)),rewards,torch.tensor(numpy.array(actions)),torch.tensor(logprobs)

def CalculateAdvantage(model,observations,RTG):

    V = model.Critic(observations)[0].detach().squeeze()
    Advantage = (torch.tensor(RTG)-V)
    return Advantage

def getRatio(model,observations,actions,old_logprobs):
    out_mean,out_variance = model.Actor(observations)
    dist = Normal(out_mean, out_variance)
    log_probs = dist.log_prob(actions).squeeze()
    return torch.exp(log_probs-old_logprobs)


def PPO(batchSize,model,rollout=10,episodes =200,clipEpsilon=0.2,df = 0.01):
    batch_size = batchSize
    rollout_size = rollout
    episode_size= episodes
    epsilon= clipEpsilon
    discountFactor = df
    policy = model
    env=gym.make("Pendulum-v1-custom")
    for i in range(batch_size):

        observations,rewards,actions,logprobs = DoRollout(policy,rollout_size,episode_size,discountFactor)
        RTG = []
        for j in range(rollout_size):
            RTG+=RewardsToGo(rewards[j],discountFactor)
        print(sum(RTG))
        Advantage = CalculateAdvantage(policy,observations,RTG)
        Advantage = (Advantage - Advantage.mean()) / (Advantage.std() + 1e-10)

        for j in range(epochs):
            ratio = getRatio(policy,observations,actions,logprobs)
            surr1 =  Advantage*ratio
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * Advantage
            actorGrad =(-torch.min(surr1, surr2)).mean()
            policy.optimActor.zero_grad()
            actorGrad.backward(retain_graph=True)
            policy.optimActor.step()

            V = policy.Critic(observations)[0].squeeze()
            criticGrad = nn.MSELoss()(V, torch.tensor(RTG))
            policy.optimCritic.zero_grad()
            criticGrad.backward()
            policy.optimCritic.step()

        policy.saveModels() 

    policy.saveModels()
    print("Saved!")

def TryModel(model):
    env=gym.make("Pendulum-v1-custom")
    obs = env.reset()
    rewards=[]
    # print(obs)
    minReward = -1000
    minobs = obs
    # print(obs)
    for i in range(1000):
        action,_ = model(torch.tensor(obs))
        obs, reward, done, info = env.step(action.detach())

        if reward > minReward:
            minReward=reward
            minobs = obs
        rewards.append(reward.tolist())
    return rewards,minReward,minobs

def EnvironmentIterationLoop(batch_size,LearningAlg,Rollout=10,episodes =200,clipEpsilon=0.2,df = 0.01):
    # if LearningAlg == "Vanilla":        
    #     Model = VanillaModel()
    #     return VanillaPolicyGradience(batch_size,Model)
    if LearningAlg == "PPO":
        Model = PPOModels()
        return PPO(batch_size,Model,Rollout,episodes,clipEpsilon,df)
    # elif LearningAlg == "Critic":
    #     Model = VanillaModel()
    #     return LearnCritic(batch_size,Model)
    else:
        print("No Option Selected")
        return


#EnvironmentIterationLoop(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),float(sys.argv[5]),float(sys.argv[6]))
model = PendulumNN(3,1)
model.load_state_dict(torch.load("PPOClip.pt"))
x,b,f = TryModel(model)
model2 = PastPendulumNN(3,1)

y,c,g = TryModel(model2)
print(sum(x))
print(b)
print(f)
print(sum(y))
print(c)
print(g)

#print(numpy.array([[1,2,3],[4,5,6]]).sum(axis=0))
