'''
    1. Don't delete anything which is already there in code.
    2. you can create your helper functions to solve the task and call them.
    3. Don't change the name of already existing functions.
    4. Don't change the argument of any function.
    5. Don't import any other python modules.
    6. Find in-line function comments.

'''

import gym
import numpy as np
import math
import time
import argparse
import matplotlib.pyplot as plt


class sarsaAgent():
    '''
    - constructor: graded
    - Don't change the argument of constructor.
    - You need to initialize epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2 and weight_T1, weights_T2 for task-1 and task-2 respectively.
    - Use constant values for epsilon_T1, epsilon_T2, learning_rate_T1, learning_rate_T2.
    - You can add more instance variable if you feel like.
    - upper bound and lower bound are for the state (position, velocity).
    - Don't change the number of training and testing episodes.
    '''

    def __init__(self):
        self.step_T1=18
        self.step_T2=6
        self.tiles=6
        self.env = gym.make('MountainCar-v0')
        self.epsilon_T1 = 0.01
        self.epsilon_T2 = 0.01
        self.learning_rate_T1 = 0.05
        self.learning_rate_T2 = 0.01
        self.weights_T1 = np.random.rand(self.step_T1,self.step_T1,3,1)
        self.weights_T1[-1]=0
        self.weights_T2 = np.random.rand(self.step_T2,self.step_T2,3,self.tiles)
        self.discount = 1.0
        self.train_num_episodes = 10000
        self.test_num_episodes = 100
        self.upper_bounds = np.array([self.env.observation_space.high[0], self.env.observation_space.high[1]])
        self.lower_bounds = np.array([self.env.observation_space.low[0], self.env.observation_space.low[1]])

        self.step_size_T1=(self.upper_bounds-self.lower_bounds)/self.step_T1

        self.step_size_T2=(self.upper_bounds-self.lower_bounds)*6/31
        self.tile_size_T2=self.step_size_T2/6

    '''
    - get_table_features: Graded
    - Use this function to solve the Task-1
    - It should return representation of state.
    '''

    def get_table_features(self, obs):
        a=np.int64((obs-self.lower_bounds)/self.step_size_T1)
        return np.array([[a[0],a[1]]])

    '''
    - get_better_features: Graded
    - Use this function to solve the Task-2
    - It should return representation of state.
    '''

    def get_better_features(self, obs):
        index=[]

        for n in range(self.tiles):
            low=self.lower_bounds-n*self.tile_size_T2
            a=np.int64((obs-low)/self.step_size_T2)
            index.append([a[0],a[1]])
        index=np.array(index)
        return index

    '''
    - choose_action: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function should return a valid action.
    - state representation, weights, epsilon are set according to the task. you need not worry about that.
    '''

    def choose_action(self, state, weights, epsilon):
        roll=np.random.random()

        if roll<epsilon:
            return np.random.randint(3)

        else:
            w=np.sum(weights[state[:,0],state[:,1],:,range(state.shape[0])],axis=0)
            return np.argmax(w)

    '''
    - sarsa_update: Graded.
    - Implement this function in such a way that it will be common for both task-1 and task-2.
    - This function will return the updated weights.
    - use sarsa(0) update as taught in class.
    - state representation, new state representation, weights, learning rate are set according to the task i.e. task-1 or task-2.
    '''

    def sarsa_update(self, state, action, reward, new_state, new_action, learning_rate, weights):
        q=np.sum(weights[state[:,0],state[:,1],action,range(state.shape[0])])

        if state.shape[0]==1:
            if state[0][0] >= self.get_table_features(np.array([0.5,0]))[0][0]:
                #weights[state[0]][state[1]][action]=0
                weights[state[:,0],state[:,1],action,range(state.shape[0])]=0
                return weights

        if state.shape[0]==self.tiles:
            if state[-1][0] >= self.get_better_features(np.array([0.5,0]))[-1][0]:
                weights[state[:,0],state[:,1],action,range(state.shape[0])]+=learning_rate*(reward-q)
                return weights

        n_q=np.sum(weights[new_state[:,0],new_state[:,1],new_action,range(1)])

        weights[state[:,0],state[:,1],action,range(state.shape[0])]+=learning_rate*(reward+n_q-q)*np.ones(state.shape[0])
        return weights

    '''
    - train: Ungraded.
    - Don't change anything in this function.
    
    '''

    def train(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
            weights = self.weights_T1
            epsilon = self.epsilon_T1
            learning_rate = self.learning_rate_T1
        else:
            get_features = self.get_better_features
            weights = self.weights_T2
            epsilon = self.epsilon_T2
            learning_rate = self.learning_rate_T2
        reward_list = []
        plt.clf()
        plt.cla()
        for e in range(self.train_num_episodes):
            print(e)
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            new_action = self.choose_action(current_state, weights, epsilon)
            while not done:
                action = new_action
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                new_action = self.choose_action(new_state, weights, epsilon)
                weights = self.sarsa_update(current_state, action, reward, new_state, new_action, learning_rate,
                                            weights)
                current_state = new_state
                if done:
                    reward_list.append(-t)
                    break
                t += 1
        self.weight_T1=weights
        self.weights_T2=weights
        self.save_data(task)
        reward_list=[np.mean(reward_list[i-100:i]) for i in range(100,len(reward_list))]
        plt.plot(reward_list)
        plt.savefig(task + '.jpg')

    '''
       - load_data: Ungraded.
       - Don't change anything in this function.
    '''

    def load_data(self, task):
        return np.load(task + '.npy')

    '''
       - save_data: Ungraded.
       - Don't change anything in this function.
    '''

    def save_data(self, task):
        if (task == 'T1'):
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T1)
            f.close()
        else:
            with open(task + '.npy', 'wb') as f:
                np.save(f, self.weights_T2)
            f.close()

    '''
    - test: Ungraded.
    - Don't change anything in this function.
    '''

    def test(self, task='T1'):
        if (task == 'T1'):
            get_features = self.get_table_features
        else:
            get_features = self.get_better_features
        weights = self.load_data(task)
        reward_list = []
        for e in range(self.test_num_episodes):
            current_state = get_features(self.env.reset())
            done = False
            t = 0
            while not done:
                action = self.choose_action(current_state, weights, 0)
                obs, reward, done, _ = self.env.step(action)
                new_state = get_features(obs)
                current_state = new_state
                if done:
                    reward_list.append(-1.0 * t)
                    break
                t += 1
        return float(np.mean(reward_list))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
       help="first operand", choices={"T1", "T2"})
    ap.add_argument("--train", required=True,
       help="second operand", choices={"0", "1"})
    args = vars(ap.parse_args())
    task=args['task']
    train=int(args['train'])
    agent = sarsaAgent()
    agent.env.seed(0)
    np.random.seed(0)
    agent.env.action_space.seed(0)
    if(train):
        agent.train(task)
    else:
        print(agent.test(task))
