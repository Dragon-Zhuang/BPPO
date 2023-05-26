import gym
import d4rl
import torch
import numpy as np
import os
import time
from tqdm import tqdm
import argparse
from tensorboardX import SummaryWriter

from buffer import OfflineReplayBuffer
from critic import ValueLearner, QPiLearner, QSarsaLearner
from bppo import BehaviorCloning, BehaviorProximalPolicyOptimization


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--env", default="hopper-medium-v2")        
    parser.add_argument("--seed", default=8, type=int)
    parser.add_argument("--gpu", default=0, type=int)             
    parser.add_argument("--log_freq", default=int(2e3), type=int)
    parser.add_argument("--path", default="logs", type=str)
    # For Value
    parser.add_argument("--v_steps", default=int(2e6), type=int) 
    parser.add_argument("--v_hidden_dim", default=512, type=int)
    parser.add_argument("--v_depth", default=3, type=int)
    parser.add_argument("--v_lr", default=1e-4, type=float)
    parser.add_argument("--v_batch_size", default=512, type=int)
    # For Q
    parser.add_argument("--q_bc_steps", default=int(2e6), type=int) 
    parser.add_argument("--q_pi_steps", default=10, type=int) 
    parser.add_argument("--q_hidden_dim", default=1024, type=int)
    parser.add_argument("--q_depth", default=2, type=int)       
    parser.add_argument("--q_lr", default=1e-4, type=float) 
    parser.add_argument("--q_batch_size", default=512, type=int)
    parser.add_argument("--target_update_freq", default=2, type=int)
    parser.add_argument("--tau", default=0.05, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--is_offpolicy_update", default=False, type=bool)
    # For BehaviorCloning
    parser.add_argument("--bc_steps", default=int(5e5), type=int) # try to reduce the bc step if it works poorly, 1e-5 for example
    parser.add_argument("--bc_hidden_dim", default=1024, type=int)
    parser.add_argument("--bc_depth", default=2, type=int)
    parser.add_argument("--bc_lr", default=1e-4, type=float)
    parser.add_argument("--bc_batch_size", default=512, type=int)
    # For BPPO 
    parser.add_argument("--bppo_steps", default=int(1e3), type=int)
    parser.add_argument("--bppo_hidden_dim", default=1024, type=int)
    parser.add_argument("--bppo_depth", default=2, type=int)
    parser.add_argument("--bppo_lr", default=1e-4, type=float)  
    parser.add_argument("--bppo_batch_size", default=512, type=int)
    parser.add_argument("--clip_ratio", default=0.25, type=float)
    parser.add_argument("--entropy_weight", default=0.0, type=float) # for ()-medium-() tasks, try to use the entropy loss, weight == 0.01
    parser.add_argument("--decay", default=0.96, type=float)
    parser.add_argument("--omega", default=0.9, type=float)
    parser.add_argument("--is_clip_decay", default=True, type=bool)  
    parser.add_argument("--is_bppo_lr_decay", default=True, type=bool)       
    parser.add_argument("--is_update_old_policy", default=True, type=bool)
    parser.add_argument("--is_state_norm", default=True, type=bool)
    
    args = parser.parse_args()
    print(f'------current env {args.env} and current seed {args.seed}------')
    # path
    current_time = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    path = os.path.join(args.path, args.env, str(args.seed))
    os.makedirs(os.path.join(path, current_time))
    # save args
    config_path = os.path.join(path, current_time, 'config.txt')
    config = vars(args)
    with open(config_path, 'w') as f:
        for k, v in config.items():
            f.writelines(f"{k:20} : {v} \n")


    env = gym.make(args.env)
    # seed
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    # dim of state and action
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    

    # offline dataset to replay buffer
    dataset = env.get_dataset()
    replay_buffer = OfflineReplayBuffer(device, state_dim, action_dim, len(dataset['actions']))
    replay_buffer.load_dataset(dataset=dataset)
    replay_buffer.compute_return(args.gamma)
    
    #for hopper-medium-v2 task, don't use state normalize
    if args.is_state_norm:
        mean, std = replay_buffer.normalize_state()
    else:
        mean, std = 0., 1.

    # summarywriter logger
    comment = args.env + '_' + str(args.seed)
    logger_path = os.path.join(path, current_time)
    logger = SummaryWriter(log_dir=logger_path, comment=comment)
    
    
    # initilize
    value = ValueLearner(device, state_dim, args.v_hidden_dim, args.v_depth, args.v_lr, args.v_batch_size)
    Q_bc = QSarsaLearner(device, state_dim, action_dim, args.q_hidden_dim, args.q_depth, args.q_lr, args.target_update_freq, args.tau, args.gamma, args.q_batch_size)
    if args.is_offpolicy_update:
        Q_pi = QPiLearner(device, state_dim, action_dim, args.q_hidden_dim, args.q_depth, args.q_lr, args.target_update_freq, args.tau, args.gamma, args.q_batch_size)
    bc = BehaviorCloning(device, state_dim, args.bc_hidden_dim, args.bc_depth, action_dim, args.bc_lr, args.bc_batch_size)
    bppo = BehaviorProximalPolicyOptimization(device, state_dim, args.bppo_hidden_dim, args.bppo_depth, action_dim, args.bppo_lr, args.clip_ratio, args.entropy_weight, args.decay, args.omega, args.bppo_batch_size)


    # value training 
    value_path = os.path.join(path, 'value.pt')
    if os.path.exists(value_path):
        value.load(value_path)
    else:
        for step in tqdm(range(int(args.v_steps)), desc='value updating ......'): 
            value_loss = value.update(replay_buffer)
            
            if step % int(args.log_freq) == 0:
                print(f"Step: {step}, Loss: {value_loss:.4f}")
                logger.add_scalar('value_loss', value_loss, global_step=(step+1))

        value.save(value_path)

    # Q_bc training
    Q_bc_path = os.path.join(path, 'Q_bc.pt')
    if os.path.exists(Q_bc_path):
        Q_bc.load(Q_bc_path)
    else:
        for step in tqdm(range(int(args.q_bc_steps)), desc='Q_bc updating ......'): 
            Q_bc_loss = Q_bc.update(replay_buffer, pi=None)

            if step % int(args.log_freq) == 0:
                print(f"Step: {step}, Loss: {Q_bc_loss:.4f}")
                logger.add_scalar('Q_bc_loss', Q_bc_loss, global_step=(step+1))

        Q_bc.save(Q_bc_path)
    
    if args.is_offpolicy_update:
        Q_pi.load(Q_bc_path)

    # bc training
    best_bc_path = os.path.join(path, 'bc_best.pt')
    if os.path.exists(best_bc_path):
        bc.load(best_bc_path)
    else:
        best_bc_score = 0    
        for step in tqdm(range(int(args.bc_steps)), desc='bc updating ......'):
            bc_loss = bc.update(replay_buffer)

            if step % int(args.log_freq) == 0:
                current_bc_score = bc.offline_evaluate(args.env, args.seed, mean, std)
                if current_bc_score > best_bc_score:
                    best_bc_score = current_bc_score
                    bc.save(best_bc_path)
                    np.savetxt(os.path.join(path, 'best_bc.csv'), [best_bc_score], fmt='%f', delimiter=',')
                print(f"Step: {step}, Loss: {bc_loss:.4f}, Score: {current_bc_score:.4f}")
                logger.add_scalar('bc_loss', bc_loss, global_step=(step+1))
                logger.add_scalar('bc_score', current_bc_score, global_step=(step+1))

        bc.save(os.path.join(path, 'bc_last.pt'))
        bc.load(best_bc_path)


    # bppo training    
    bppo.load(best_bc_path)
    bppo.set_old_policy(bc._policy)
    best_bppo_path = os.path.join(path, current_time, 'bppo_best.pt')
    Q = Q_bc

    best_bppo_score = bppo.offline_evaluate(args.env, args.seed, mean, std)
    print('best_bppo_score:',best_bppo_score,'-------------------------')

    for step in tqdm(range(int(args.bppo_steps)), desc='bppo updating ......'):
        if step > 200:
            args.is_clip_decay = False
            args.is_bppo_lr_decay = False
        bppo_loss = bppo.update(replay_buffer, Q, value, args.is_clip_decay, args.is_bppo_lr_decay)
        current_bppo_score = bppo.offline_evaluate(args.env, args.seed, mean, std)

        if current_bppo_score > best_bppo_score:
            best_bppo_score = current_bppo_score
            print('best_bppo_score:',best_bppo_score,'-------------------------')
            bppo.save(best_bppo_path)
            np.savetxt(os.path.join(path, current_time, 'best_bppo.csv'), [best_bppo_score], fmt='%f', delimiter=',')

            if args.is_update_old_policy:
                bppo.set_old_policy(bppo._policy)

        if args.is_offpolicy_update:
            for _ in tqdm(range(int(args.q_pi_steps)), desc='Q_pi updating ......'): 
                Q_pi_loss = Q_pi.update(replay_buffer, bppo)

            Q = Q_pi

        print(f"Step: {step}, Loss: {bppo_loss:.4f}, Score: {current_bppo_score:.4f}")
        logger.add_scalar('bppo_loss', bppo_loss, global_step=(step+1))
        logger.add_scalar('bppo_score', current_bppo_score, global_step=(step+1))
    
    logger.close()
