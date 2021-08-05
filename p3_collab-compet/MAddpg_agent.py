import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_agent import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.01             # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0.00000  # L2 weight decay
UPDATE_EVERY = 1      # how often to update the network
LEARN_STEPS = 1        # how many times to learn each avtive step
NOISE_DECAY = 0.9999    # noise decay rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agent, random_seed):
        """Initialize an MADDPG_Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agent (int): number of DDPG agents

        """
        # Create internal number of agent variable
        self.num_agent = num_agent
        
        # cCreate DDPG agents object
        self.DDPG_agents = [Agent(state_size, action_size, 
                                  random_seed, LR_ACTOR, 
                                  LR_CRITIC, WEIGHT_DECAY) for i in range(num_agent)]
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Noise process
        self.noise = [OUNoise(action_size, random_seed, NOISE_DECAY) for i in range(num_agent)]
        
        # tracking step count, will be reset to 0 when update occur
        self.t = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        
        # update internal step count
        self.t = self.t+1

        # Learn, if enough samples are available in memory and step count = UPDATE_EVERY
        # when update happens, reduce randomness scale in noise function
        if len(self.memory) > BATCH_SIZE and (self.t % UPDATE_EVERY) == 0:

            for _ in range(LEARN_STEPS):
                for agent_id in range(self.num_agent):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA, agent_id)
                
                for agent_id in range(self.num_agent):
                    self.soft_update(self.DDPG_agents[agent_id].critic_local, self.DDPG_agents[agent_id].critic_target, TAU)
                    self.soft_update(self.DDPG_agents[agent_id].actor_local, self.DDPG_agents[agent_id].actor_target, TAU)  
                    
            # reset internal step count
            self.t = 0

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        actions = []
        
        # loop through each agent
        for i in range(self.num_agent):
            state_per_agent = torch.from_numpy(state[i:i+1,:]).float().to(device)
            
            # enable evaluation mode
            self.DDPG_agents[i].actor_local.eval()
            with torch.no_grad():
                action_per_agent = self.DDPG_agents[i].actor_local(state_per_agent).cpu().data.numpy()
            
            # switch back to train mode
            self.DDPG_agents[i].actor_local.train()
            
            # add noise to actions
            if add_noise:
                action_per_agent += self.noise[i].sample()
                
            # clamp actions to the value between -1 and 1
            action_per_agent = np.clip(action_per_agent, -1.0, 1.0)
            
            # insert actions to list
            actions.append(action_per_agent)
        
        # concat list to create variable with dimension = (num_agents, action_size)
        actions = np.concatenate(tuple(actions), 0)
        
        # update scale in noise funtion
        for noise_obj in self.noise:
            noise_obj.scale = max(noise_obj.scale*NOISE_DECAY, 0.01)
                  
        return actions

    def reset(self):
        for i in range(self.num_agent):
            self.noise[i].reset()

    def learn(self, experiences, gamma, agent_id):
        """Update policy and value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        # break tuple object into states, actions, rewards, next_states, dones numpy array
        states, actions, rewards, next_states, dones = experiences
        
        # calculate next_actions_all and pred_actions_all from all agents
        next_actions = torch.zeros_like(actions)
#         next_actions_all = []
        for i in range(self.num_agent):
            # retrieve state for actor network
            next_actions[:,i,:] = self.DDPG_agents[i].actor_target(next_states[:,i,:])
#             next_states_actor = next_states[:,i,:]
            
            # calculate next_action and pred_action
#             next_action = self.DDPG_agents[i].actor_target(next_states_actor)
            
            # transpose then append next_action and pred_action to list
            # final list dimension becomes (action_size*num_agents, samples)
#             next_actions_all.append(next_action.transpose(0,1))
        
        # transfer list to tensor and transpose tensor back to dimension (samples, action_size*num_agents)
#         next_actions_all = torch.transpose(torch.cat(tuple(next_actions_all), 0), 0, 1)
        d = next_actions.size()
        next_actions_all = torch.reshape(next_actions,(d[0], d[1]*d[2]))
        
        # reshape states and actions to get states_all and actions_all for critic
        # change shape from (samples, # of agents, state_size) to (samples, (# of agents)*(state_size))
        d = states.size()
        states_all = torch.reshape(states,(d[0], d[1]*d[2]))
        next_states_all = torch.reshape(next_states,(d[0], d[1]*d[2]))
        
        # reshape states and actions to get states_all and actions_all for critic
        # change shape from (samples, # of agents, action_size) to (samples, (# of agents)*(action_size))
        d = actions.size()
        actions_all = torch.reshape(actions,(d[0], d[1]*d[2]))
        
#         #　loop through each agent and optimize it
#         for i in range(self.num_agent):
            
        # rerieve states_per_agent
#         states_per_agent = states[:,agent_id,:]

        # retrieve reward for individual DDPG agent 
        # (indices "i:i+1" will keep the last dimension)
        rewards_per_agent = rewards[:,agent_id:agent_id+1]

        # retrieve dones for each agent
        dones_per_agent = dones[:,agent_id:agent_id+1]

        # calculate predict action from curret states
        # if the agent used to calculate pred_action is not the agent under training, detach pred_action
        pred_actions = torch.zeros_like(actions)
#         pred_actions_all = []
        for j in range(self.num_agent):
            if j == agent_id:
                pred_actions[:,j,:] = self.DDPG_agents[j].actor_local(states[:,j,:])
            else:
#                 pred_action = self.DDPG_agents[j].actor_local(states[:,j,:])
#                 pred_actions[:,j,:] = self.DDPG_agents[j].actor_local(states[:,j,:]).detach()
                pred_actions[:,j,:] = actions[:,j,:]

            # transpose pred_action dimension from (samples, action_size) to (action_size, samples)
            # then insert trasposed(pred_action) into list
#             pred_actions_all.append(torch.transpose(pred_action, 0, 1))

        # concate list of dimension (num_agent,) to (action_size*2, samples)
#         pred_actions_all = torch.cat(tuple(pred_actions_all), 0)

        # transpose from (action_size*2, samples) to (samples, action_size*2)
#         pred_actions_all = torch.transpose(pred_actions_all, 0, 1)
        d = pred_actions.size()
        pred_actions_all = torch.reshape(pred_actions,(d[0], d[1]*d[2]))

        # ---------------------------- calculate critic loss---------------------------- #
        Q_targets_next = self.DDPG_agents[agent_id].critic_target(next_states_all, next_actions_all)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards_per_agent + (gamma * Q_targets_next * (1 - dones_per_agent))
        # Compute critic loss
        Q_expected = self.DDPG_agents[agent_id].critic_local(states_all, actions_all)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize critic loss
        self.DDPG_agents[agent_id].critic_optimizer.zero_grad()
        critic_loss.backward()
        self.DDPG_agents[agent_id].critic_optimizer.step()

        # ---------------------------- calculate actor loss---------------------------- #
        # Compute actor loss
        actor_loss = -self.DDPG_agents[agent_id].critic_local(states_all, pred_actions_all).mean()
        # Minimize actor loss
        self.DDPG_agents[agent_id].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.DDPG_agents[agent_id].actor_optimizer.step()    

#         # ----------------------- update target networks ----------------------- #
#         self.soft_update(self.DDPG_agents[agent_id].critic_local, self.DDPG_agents[agent_id].critic_target, TAU)
#         self.soft_update(self.DDPG_agents[agent_id].actor_local, self.DDPG_agents[agent_id].actor_target, TAU)    
            
            

#         # ---------------------------- update critic ---------------------------- #
#         # Get predicted next-state actions and Q values from target models
#         actions_next = self.actor_target(next_states)
#         Q_targets_next = self.critic_target(next_states, actions_next)
#         # Compute Q targets for current states (y_i)
#         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
#         # Compute critic loss
#         Q_expected = self.critic_local(states, actions)
#         critic_loss = F.mse_loss(Q_expected, Q_targets)
#         # Minimize the loss
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         # ---------------------------- update actor ---------------------------- #
#         # Compute actor loss
#         actions_pred = self.actor_local(states)
#         actor_loss = -self.critic_local(states, actions_pred).mean()
#         # Minimize the loss
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # ----------------------- update target networks ----------------------- #
#         self.soft_update(self.critic_local, self.critic_target, TAU)
#         self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, NOISE_DECAY, mu=0., theta=0.15, sigma=0.2, scale=1.0):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.scale = scale
        self.decay = NOISE_DECAY
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
#         self.scale = max(self.scale*self.decay,0.01)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.standard_normal() for i in range(len(x))])
        self.state = x + dx
#         self.state = self.sigma * np.array([np.random.standard_normal() for i in range(self.size)])
        return self.state*self.scale

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # form numpy array for states, actions, rewards, next_states, and dones
        states = np.array([e.state for e in experiences if e is not None])       
        actions = np.array([e.action for e in experiences if e is not None])
        rewards = np.array([e.reward for e in experiences if e is not None])
        next_states = np.array([e.next_state for e in experiences if e is not None])
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8)
        
        # transfer above variables to tensors
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)