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
        self.optimActor = Adam(self.Actor.parameters(), lr=0.1)
        self.optimCritic = Adam(self.Critic.parameters(), lr=0.1)

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
    return [rewards[0]+0.9*RewardNext[0]]+RewardNext


def DoRollout(Models,rollout_size,episode_size,discountFactor):
    obervations =[]
    rewards = []
    actions = []
    logprobs = []
    for i in range(rollout_size):
        episodeObs =[]
        episodeRewards = []
        episodeActs = []
        episodeLogprobs = []
        obs = env.reset()
        for j in range(episode_size):
            out_mean,out_variance   = Models.Actor(torch.as_tensor(obs))
            out_action_distribution = Normal(out_mean, out_variance)
            action                  = out_action_distribution.sample()
            obs, reward, done, info = env.step(action)
            logprob = out_action_distribution.log_prob(action)
            episodeObs.append(obs)
            episodeRewards.append(reward.tolist())
            episodeActs.append(action)
            episodeLogprobs.append(logprob)
        obervations.append(episodeObs)
        rewards.append(episodeRewards)
        actions.append(episodeActs)
        logprobs.append(episodeLogprobs)
    return obervations,rewards,actions,logprobs
        
def calculateAdvantage(Models,observations,rewards):
    CriticValue,_ = Models.Critic(observations)
    return torch.subtract(CriticValue,CriticValue)

def getRatio(Model,oldProbs,observations,actions):
    out_mean,out_variance = Model.Actor(observations)
    out_action_distribution = Normal(out_mean, out_variance)
    NewProbs = out_action_distribution.log_prob(actions)
    return torch.exp(NewProbs-oldProbs)

def getGradient(logprobs,npRTG,rollout_size,episode_size):
    # prob = logprobs[0][0]
    # for i in range(1,episode_size):
    #     prob+=logprobs[0][i]
    # trajLogProbs= prob
    # for i in range(1,rollout_size):
    #     prob = logprobs[i][0]
    #     for k in range(1,episode_size):
    #         prob+=logprobs[i][k]
    #     trajLogProbs+= prob
    # return trajLogProbs
    # prob = logprobs[0][0]*-npRTG[0][0]
    # for i in range(1,episode_size):
    #     prob+=logprobs[0][i]*-npRTG[0][i]
    # trajLogProbs= prob
    # for i in range(1,rollout_size):
    #     prob = logprobs[i][0]*-npRTG[i][0]
    #     for k in range(1,episode_size):
    #         prob+=logprobs[i][k]*-npRTG[i][k]
    #     trajLogProbs+= prob

    prob = logprobs[0][0]#*-npRTG[0][0]
    # for i in range(1,episode_size):
    #     prob+=logprobs[0][i]#*-npRTG[0][i]
    trajLogProbs= prob*-npRTG[0]
    for i in range(1,rollout_size):
        prob = logprobs[i][0]#*-npRTG[i][0]
        # for k in range(1,episode_size):
        #     prob+=logprobs[i][k]#*-npRTG[i][k]
        trajLogProbs+= prob*-npRTG[i]
    return trajLogProbs/rollout_size
    
def CriticDoRollout(Models,rollout_size,episode_size,discountFactor):
    obervations =[]
    rewards = []
    actions = []
    logprobs = []
    for i in range(rollout_size):
        episodeObs =[]
        episodeRewards = []
        episodeActs = []
        episodeLogprobs = []
        obs = env.reset()
        for j in range(episode_size):
            out_mean,out_variance   = Models.Actor(torch.as_tensor(obs))
            out_action_distribution = Normal(out_mean, out_variance)
            action                  = out_action_distribution.sample()
            obs, reward, done, info = env.step(action)
            logprob = out_action_distribution.log_prob(action)
            episodeObs.append(obs)
            episodeRewards.append(reward.tolist())
            episodeActs.append(action)
            episodeLogprobs.append(logprob)
        obervations.append(episodeObs)
        rewards.append(episodeRewards)
        actions.append(episodeActs)
        logprobs.append(episodeLogprobs)
    return obervations,rewards,actions,logprobs

def LearnCritic(batchSize,model):
    batch_size = batchSize
    rollout_size = 50
    episode_size= 1
    discountFactor = 0.1
    policy = model
    score =[]
    env=gym.make("Pendulum-v1-custom")
    for i in range(batch_size):
        observations,rewards,actions,logprobs = DoRollout(policy,rollout_size,episode_size,discountFactor)
        trajectoryGrads =[]
        print(len(logprobs))
        return
        # grad = logprobs.spread()
        # policy.optimActor.zero_grad()
        # grad.mean().backward()
        # policy.optimActor.step()
        # if i %10==1:

# def VanillaPolicyGradience(batchSize,model):
#     # if batchSize < 1000:
#     #     return "Error, batchsize too small"
#     batch_size = batchSize
#     rollout_size = 50
#     episode_size= 200
#     discountFactor = 0.1
#     policy = model
#     score =[]
#     env=gym.make("Pendulum-v1-custom")
#     for i in range(batch_size):
#         observations,rewards,actions,logprobs = DoRollout(policy,rollout_size,episode_size,discountFactor)
#         trajectoryGrads =[]
#         RewardsToGos =[]
#         for j in range(rollout_size): RewardsToGos.append(RewardsToGo(rewards[j],discountFactor)[0])
#         npRTG= numpy.array(RewardsToGos)
#         score.append(npRTG.sum())
#         print(npRTG.sum())
#         meannpRTG= npRTG.mean(axis=0)
#         npRTG -=meannpRTG
#         # npRTG = npRTG.sum(axis=1)
#         grad = getGradient(logprobs,npRTG,rollout_size,episode_size)
#         policy.optimActor.zero_grad()
#         grad.mean().backward()
#         policy.optimActor.step()
#         if i %10==1:
#             print("Saved!")
#             policy.saveModels()
#     policy.saveModels()
#     return score


    #     advantage = calculateAdvantage(Model,observations,rewards)
    #     for j in range(3):
    #         ratio = getRatio(Model,logprobs,observations,actions)
    #         clippedRatio = torch.clip(ratio,1-clipEpsilon,1+clipEpsilon)

            # ActorGrad= torch.min(ratio*advantage,clippedRatio*advantage)
            # Model.optimActor.zero_grad()
            # ActorGrad.mean().backward()
            # Model.optimActor.step()

    # Model.saveModels()


def EnvironmentIterationLoop(batch_size,LearningAlg,Critic=False,Clip=False):
    Model = VanillaModel()
    return LearnCritic(batch_size,Model)
    # if LearningAlg == "Vanilla":        
    #     Model = VanillaModel()
    #     return VanillaPolicyGradience(batch_size,Model)
    # elif LearningAlg == "PPO":
    #     Model = PPOModels()
    #     return PPO(batch_size,Model,Critic,Clip)
    # else:
    #     print("No Option Selected")
    #     return


# EnvironmentIterationLoop(10000)
def TryModel(model):
    env=gym.make("Pendulum-v1-custom")
    obs = env.reset()
    rewards=[]
    # print(obs)
    minReward = -1000
    minobs = obs
    # print(obs)
    for i in range(10000):
        action,_ = model(torch.tensor(obs))
        # print("OBS:")
        # print(obs)
        # print("Action:")
        # print(action)
        obs, reward, done, info = env.step(action.detach())
        # print("Reward:")
        # print(reward)

        if i%1==0:
            print(obs)
            print(action[0])
            print(sum(rewards))
            input()

        if reward > minReward:
            minReward=reward
            minobs = obs
        rewards.append(reward.tolist())
        # input()
    return rewards,minReward,obs

def ManualLabor():
    env=gym.make("Pendulum-v1-custom")
    obs = env.reset()
    rewards=[]
    # print(obs)
    minReward = -1000
    for i in range(100):
        action = float(input())
        obs, reward, done, info = env.step([action])
        # print(obs)
        # print(reward)
        if reward > minReward:
            minReward=reward
            input()
        rewards.append(reward.tolist())
    return rewards


score = EnvironmentIterationLoop(int(sys.argv[1]),"Critic")
# df = pd.DataFrame(numpy.array(score))
# df.to_csv("score.csv")
# model = PendulumNN(3,1)
# model.load_state_dict(torch.load("VanillaModel.pt"))
# x,b,f = TryModel(model)
# model2 = PastPendulumNN(3,1)
# model2.load_state_dict(torch.load("Best.pt"))

# y,c,g = TryModel(model2)
# print(sum(x))
# print(b)
# print(f)
# print(sum(y))
# print(c)
# print(g)
# # ManualLabor()

# print(numpy.array([[1,2,3],[4,5,6]]).sum(axis=0))
