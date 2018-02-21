from tiles3 import IHT,tiles
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import pearsonr
from matplotlib.colors import LogNorm
from matplotlib import cm
from copy import deepcopy

DORA,SOFTMAX,EGREEDY = 1,2,3

EXIT = -100

sigmoid = lambda x: np.exp(x)/(1+np.exp(x))
sigmoid_deriv = lambda x: sigmoid(x)*(1-sigmoid(x))

agents = {"softmax":SOFTMAX, "egreedy":EGREEDY, "dora":DORA}
activations = {"exp":(np.exp, np.exp), "sigmoid":(sigmoid,sigmoid_deriv)}


def process_reward(r,d,binreward):
    if binreward:
        return 1 if d else 0
    else:
        return r

def sap2tile(s,a,iht,num_tiles,max_size,scaling_factor,readonly=False):
    feature_vec = np.zeros(max_size)
    idxs = tiles(iht, num_tiles, scaling_factor*s , [a],readonly=readonly)
    feature_vec[idxs] = 1
    return feature_vec

def action_selection(q_vals, e_vals, agent, n_actions, temp, epsilon):

    if agent==DORA:
        assert(np.all(e_vals)>0)
        scores = q_vals - temp*np.log(-np.log(e_vals))
        return np.random.choice(np.flatnonzero(scores == scores.max()))
    
    elif agent==SOFTMAX:

        ps = np.exp(q_vals/temp)
        Z = np.sum(ps)
        return np.where(np.random.multinomial(1,ps/Z))[0][0]

    elif agent==EGREEDY:
        if np.random.rand(1)<epsilon:
            return np.random.randint(0,n_actions)
        else:
            return np.argmax(q_vals)

def get_bins(dim, bins_num, env):
    
    return np.linspace(env.observation_space.low[dim], env.observation_space.high[dim], bins_num + 1)


def generate_heatmap(w_Q, w_E,f, env, iht, max_size, num_tiles, scaling_factor, sample=100, dims=(100,100)):
    
    assert(len(w_Q)==len(w_E)==len(iht))
    n_models = len(w_Q)


    qvals = [np.zeros(dims) for _ in range(n_models)]
    evals = [np.zeros(dims) for _ in range(n_models)]
    
    n_actions = env.action_space.n

    row_bins = get_bins(0, dims[0], env)
    col_bins = get_bins(1, dims[1], env)

    for row in range(dims[0]):
        for col in range(dims[1]):
            sampled_states = [list(x) for x in 
								zip(np.random.uniform(row_bins[row], row_bins[row + 1], sample),
									np.random.uniform(col_bins[col], col_bins[col + 1], sample)
								)
							 ]
            
            for m in range(n_models):
    
                phis = np.zeros((sample,n_actions,max_size))
                for i,s in enumerate(sampled_states):
                    for a in range(n_actions):
                        phis[i,a,:] = sap2tile(s,a,iht[m],num_tiles,max_size,scaling_factor)
                

                qs = np.einsum('ijk,k->ij',phis,w_Q[m])
                qs = np.max(qs,1)
               

                es = np.einsum('ijk,k->ij',phis,w_E[m])
                es = f(es)
                es = np.prod(es,1)

                qvals[m][row,col] = np.mean(qs)
                evals[m][row,col] = np.mean(es)


    return qvals,evals

def get_evals(phi,w,f):
    return f(phi@w)

def run_tests(env,binreward,n_test,max_epLength,w_Q,w_E,f,iht,num_tiles,max_size,scaling_factor):
    
    n_actions = env.action_space.n

    test_rewards = []
    test_lengths = []


    for i in range(n_test):
        ep_reward = 0
        s = env.reset()
        
        for j in range(max_epLength):

            phi = np.zeros((n_actions,max_size))
            for x in range(n_actions):
                phi[x,:] = sap2tile(s,x,iht,num_tiles,max_size,scaling_factor)
            
            q_vals = phi@w_Q
            e_vals = get_evals(phi,w_E,f)
            a = np.argmax(q_vals)

            s1,r,d,_ = env.step(a)
            r = process_reward(r,d,binreward)
            ep_reward += r

            if not d:
                phi1 = np.zeros((n_actions,max_size))
                for x in range(n_actions):
                    phi1[x,:] = sap2tile(s1,x,iht,num_tiles,max_size,scaling_factor)
                
                q_vals1 = phi1@w_Q
                e_vals1 = get_evals(phi1,w_E,f)

                a1 = np.argmax(q_vals1)
                s = s1
            
            else:
                break
            
        
        test_rewards.append(ep_reward)
        test_lengths.append(j)

    return test_rewards, test_lengths

def main(opts):
    print(opts)
    env = gym.make(opts.env)

    max_size = opts.maxsize
    num_tiles = opts.tiles
    iht = IHT(max_size)
    scaling_factor = num_tiles/(env.observation_space.high - env.observation_space.low)
    n_actions = env.action_space.n

    n_episodes = opts.train
    n_test = opts.test
    max_epLength = opts.eplength

    gamma = opts.gamma
    gamme_E = opts.expgamma
    
    epsilon = opts.epsilon
    temp = opts.temperature
    alpha = opts.rate/num_tiles
    alpha_E = opts.exprate/num_tiles

    agent = agents[opts.agent]
    f,dfdx = activations[opts.activation]

    augment_reward = opts.augmentReward


    vis_dims = (100,100)
    
    w_Q = .01*np.random.randn(max_size)
    w_E = np.zeros(max_size)

    tot_rewards = []
    ep_lengths = []

    row_bins = get_bins(0, vis_dims[0],env)
    col_bins = get_bins(1, vis_dims[1],env)

    visited_states = []
    taken_actions = []

    test_rewards = []
    test_lengths = []

    model_snapshot_E = []
    model_snapshot_Q = []
    model_snapshot_idxs = []
    model_snapshot_ihts = []
    

    for i in range(n_episodes):
        
        if opts.save_model_each>0 and not i%opts.save_model_each:
            model_snapshot_Q.append(np.copy(w_Q))
            model_snapshot_E.append(np.copy(w_E))
            model_snapshot_idxs.append(len(visited_states))
            model_snapshot_ihts.append(deepcopy(iht))

        if opts.testeach>0 and not i%opts.testeach:

            snapshot_rewards, snapshot_lengths = run_tests(env,opts.binreward,n_test,max_epLength,w_Q,w_E,f,deepcopy(iht),num_tiles,max_size,scaling_factor)
            
            test_lengths.append(np.mean(snapshot_lengths))
            test_rewards.append(np.mean(snapshot_rewards))
            

        ep_reward = 0
        s = env.reset()
        visited_states.append(s)

        phi = np.zeros((n_actions,max_size))
        for x in range(n_actions):
            phi[x,:] = sap2tile(s,x,iht,num_tiles,max_size,scaling_factor)
        
        q_vals = phi@w_Q
        e_vals = get_evals(phi,w_E,f)
        a = action_selection(q_vals, e_vals,agent,n_actions,temp,epsilon)

        for j in range(max_epLength):
            

            phi = np.zeros((n_actions,max_size))
            for x in range(n_actions):
                phi[x,:] = sap2tile(s,x,iht,num_tiles,max_size,scaling_factor)
            
            q_vals = phi@w_Q
            e_vals = get_evals(phi,w_E,f)

            taken_actions.append(a)
            s1,r,d,_ = env.step(a)
            visited_states.append(s1)
            r = process_reward(r,d,opts.binreward)
            ep_reward += r

            if not d:
                phi1 = np.zeros((n_actions,max_size))
                for x in range(n_actions):
                    phi1[x,:] = sap2tile(s1,x,iht,num_tiles,max_size,scaling_factor)
                
                q_vals1 = phi1@w_Q
                e_vals1 = get_evals(phi1,w_E,f)

                a1 = action_selection(q_vals1, e_vals1,agent,n_actions,temp,epsilon)

                w_E += alpha_E*(0 + gamme_E*e_vals1[a1] - e_vals[a])*phi[a,:]*dfdx(e_vals[a])
                
                if augment_reward:
                    r += 1/(np.log(get_evals(phi,w_E,f)[a])/np.log(1-alpha_E))

                w_Q += alpha*(r + gamma*np.max(q_vals1) - q_vals[a])*phi[a,:]

                s = s1
                a = a1
            
            else:
                w_E += alpha_E*(0 - e_vals[a])*phi[a,:]*e_vals[a]*(1-e_vals[a])
                
                if augment_reward:
                    r += 1/(np.log(get_evals(phi,w_E,f)[a])/np.log(1-alpha_E))

                w_Q += alpha*(r - q_vals[a])*phi[a,:]

                taken_actions.append(EXIT)

                break
                    
        if j==max_epLength-1 and not d:
            taken_actions.append(EXIT)

        if not i%10 and not opts.quiet:
            print("finished episode {0} in {2} steps with reward {1}".format(i,ep_reward,j+1))
        tot_rewards.append(ep_reward)
        ep_lengths.append(j)
    
    
    
    snapshot_rewards, snapshot_lengths = run_tests(env,opts.binreward,n_test,max_epLength,w_Q,w_E,f,deepcopy(iht),num_tiles,max_size,scaling_factor)        
    test_lengths.append(np.mean(snapshot_lengths))
    test_rewards.append(np.mean(snapshot_rewards))

    model_snapshot_Q.append(np.copy(w_Q))
    model_snapshot_E.append(np.copy(w_E))
    model_snapshot_idxs.append(len(visited_states))
    model_snapshot_ihts.append(deepcopy(iht))


    visited_states = np.asarray(visited_states)
    taken_actions = np.asarray(taken_actions).reshape((-1,1))
    saps = np.concatenate((visited_states, taken_actions), axis=1)

    record = {
        "opts": opts,
        "visited_states": visited_states,
        "test_rewards": test_rewards,
        "test_lengths": test_lengths,
        "train_rewards": tot_rewards,
        "train_lengths": ep_lengths
    }

    if opts.record:
        out_file = opts.outfile
        np.save(out_file, record)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a","--agent", help="dora, softmax or egreedy",default="egreedy")
    parser.add_argument("-e","--env", help="Enviornment to run",default="MountainCar-v0")
    parser.add_argument("--binreward", help="Set binary reward (1/0 for success/else)", action="store_true", default=False) 
    parser.add_argument("--activation", help="Non-linearity for E-values: sigmoid or exp. e=f(phi*w_E)", default="sigmoid") 
    parser.add_argument("--augmentReward",help="Use E-values as exploration bonus added to reward", action="store_true", default=False)

    parser.add_argument("--train", help="number of training episodes",default=500,type=int)
    parser.add_argument("--test", help="number of test episodes",default=10,type=int)
    parser.add_argument("--testeach", help="intervals of episodes to test",default=100,type=int)
    parser.add_argument("--save_model_each", help="intervals of episodes to save model sanpshot",default=10,type=int)
    parser.add_argument("--eplength", help="max steps in episode",default=1000,type=int)
    
    parser.add_argument("--tiles", help="number of tiles",default=8,type=int)
    parser.add_argument("--maxsize", help="max size (features)",default=2048,type=int)

    parser.add_argument("--rate", help="learning rate (will be divided by tiles)",default=0.1,type=float)
    parser.add_argument("--exprate", help="exploration learning rate (will be divided by tiles)",default=0.5,type=float)
    
    parser.add_argument("--gamma", help="discount factor",default=0.99,type=float)
    parser.add_argument("--expgamma", help="exploration discount factor",default=0.99,type=float)

    parser.add_argument("--epsilon", help="epsilon for e-greedy",default=0.3,type=float)
    parser.add_argument("--temperature", help="temperature for softmax",default=0.5,type=float)

    parser.add_argument("-q","--quiet", help="Suppress output to terminal", action="store_true", default=False)
    parser.add_argument("-r","--record", help="Record logs and data", action="store_true", default=False)
    parser.add_argument("--outfile", help="Destination for saving record",default="log.npy")


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    #print(args.env)
    main(args)
