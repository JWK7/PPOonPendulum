import gym
import a3_gym_env
import Modules
import torch_misc
env = gym.make('Pendulum-v1-custom')
import time
import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
from Modules import NormalModule
from Modules import PendulumNN
from torch_misc import vectorized_vs_nonvectorized
import numpy
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



def VanillaPolicyGradience(batchSize,model):
    # if batchSize < 1000:
    #     return "Error, batchsize too small"
    batch_size = batchSize
    rollout_size = 100
    episode_size= 1
    discountFactor = 0.1
    policy = model
    env=gym.make("Pendulum-v1-custom")
    for i in range(batch_size):
        observations,rewards,actions,logprobs = DoRollout(policy,rollout_size,episode_size,discountFactor)
        trajectoryGrads =[]
        for j in range(len(observations)):
            # print(logprobs[j])
            # print("\n\n")
            # print(RewardsToGo(rewards[j],discountFactor))
            # trajectoryGrad = logprobs[j][0]*RewardsToGo(rewards[j],discountFactor)[0]
            trajectoryGrad = logprobs[j][0]*rewards[j][0]
            # for k in range(1,episode_size):
            #     trajectoryGrad +=logprobs[j][k]*RewardsToGo(rewards[j],discountFactor)[k]
            trajectoryGrads.append(trajectoryGrad)
        FinalGrad = trajectoryGrads[0]
        for j in range(1,len(observations)):
            FinalGrad+=trajectoryGrads[j]
            # ActorGrad= trajectoryGrads[j]
            # policy.optimActor.zero_grad()
            # ActorGrad.mean().backward()
            # policy.optimActor.step()
        FinalGrad/=len(observations)
        ActorGrad= FinalGrad
        policy.optimActor.zero_grad()
        ActorGrad.mean().backward()
        policy.optimActor.step()
    policy.saveModels()


    #     advantage = calculateAdvantage(Model,observations,rewards)
    #     for j in range(3):
    #         ratio = getRatio(Model,logprobs,observations,actions)
    #         clippedRatio = torch.clip(ratio,1-clipEpsilon,1+clipEpsilon)

            # ActorGrad= torch.min(ratio*advantage,clippedRatio*advantage)
            # Model.optimActor.zero_grad()
            # ActorGrad.mean().backward()
            # Model.optimActor.step()

    # Model.saveModels()

def PPO(batchSize,Models,critic,clip):
    # if batchSize < 1000:
    #     return "Error, batchsize too small"
    batch_size = batchSize
    rollout_size = 100
    episode_size= 30
    discountFactor = 0.1
    Model = Models
    env=gym.make("Pendulum-v1-custom")
    for i in range(batch_size):
        observations,rewards,actions,logprobs = DoRollout(Model,rollout_size,episode_size,discountFactor)
        print(logprobs)
        return
        # advantage = calculateAdvantage(Model,observations,rewards)
        # for j in range(3):
        #     ratio = getRatio(Model,logprobs,observations,actions)
        #     clippedRatio = torch.clip(ratio,1-clipEpsilon,1+clipEpsilon)

        #     ActorGrad= torch.min(ratio*advantage,clippedRatio*advantage)
        #     Model.optimActor.zero_grad()
        #     ActorGrad.mean().backward()
        #     Model.optimActor.step()


            # CriticMean,CriticVar= Models.Critic(observations)
            # CriticDist = Normal(CriticMean, CriticVar)
            # sample             = CriticDist.sample()
            # logprob = CriticDist.log_prob(sample)
            # CriticGrad = logprob*(torch.subtract(torch.tensor(rewards),sample))
            # Model.optimCritic.zero_grad()
            # CriticGrad.mean().backward(retain_graph=True)
            # Model.optimCritic.step()
    Model.saveModels()

def EnvironmentIterationLoop(batch_size,LearningAlg,Critic=False,Clip=False):
    if LearningAlg == "Vanilla":        
        Model = VanillaModel()
        return VanillaPolicyGradience(batch_size,Model)
    elif LearningAlg == "PPO":
        Model = PPOModels()
        return PPO(batch_size,Model,Critic,Clip)
    else:
        print("No Option Selected")
        return


# EnvironmentIterationLoop(10000)
def TryModel(model):
    env=gym.make("Pendulum-v1-custom")
    obs = env.reset()
    rewards=[]
    # print(obs)
    for i in range(10):
        action = model(torch.tensor(obs))
        print(action[0])
        obs, reward, done, info = env.step(torch.tensor(action))
        # print(obs)
        # print(reward)
        rewards.append(reward.tolist())
    return rewards

def ManualLabor():
    env=gym.make("Pendulum-v1-custom")
    obs = env.reset()
    rewards=[]
    print(obs)
    for i in range(100):
        action = float(input())
        obs, reward, done, info = env.step([action])
        print(obs)
        print(reward)
        rewards.append(reward.tolist())
    return rewards

print(EnvironmentIterationLoop(1,"Vanilla"))
model = PendulumNN(3,1)
model.load_state_dict(torch.load("VanillaModel.pt"))
TryModel(model)
# model2 = PendulumNN(3,1)
# model2.load_state_dict(torch.load("VanillaModel2.pt"))
# x = (TryModel(model))
# y = TryModel(model2)
# print(sum(x))
# print(sum(y))
# ManualLabor()