'''Record OpenAI Baselines models interacting with OpenAI Gym environments.'''


import logging
from importlib import reload
reload(logging)  # necessary for logging in Jupyter notebooks
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(level=logging.INFO) 

import time
import numpy as np
import matplotlib.pyplot as plt

from .utils import get_activations

class Recorder(object):
    
    def __init__(self, env, model, operations):
        
        self.env = env
        self.model = model
        self.operations = operations

        self.frames = []
        self.observations = []
        self.actions = []
        self.episode_rewards = []
        
        self.n_runs = []
        self.n_steps = []
        
        self.activations = {}
        for op_name in self.operations:
            self.activations[op_name] = []
        

    def record(self, session, feed_operations, max_steps=None, max_episodes=1, sample_modulo=1):
        '''
        TODO Make feed_operations accept both Tensorflow operations and operation names
        TODO Use Tensorflow writer instead of keeping records in memory
        TODO Think about max_episodes and max_steps order
        '''
        
        logger.info(f'Start recording for {max_steps} steps or {max_episodes} episodes '
                    f'with a sample modulo of {sample_modulo}')

        n_run = 0
        total_steps = 0
        start_time = time.time()
        stop = False
        feed_dict = {}

        while n_run < max_episodes:

            n_run += 1
            n_step = 0
            ob = self.env.reset()  # LazyFrames object that stores the observation as numpy array
            ob = ob[None]  # extract Numpy array from LazyFrames object
            done = False
            episode_reward = 0

            while not done:

                total_steps += 1
                if max_steps and total_steps >= max_steps:
                    stop = True
                    break
                
                n_step += 1
                record_step = (n_step % sample_modulo == 0)
                
                if record_step:
                    self.n_runs.append(n_run)
                    self.n_steps.append(n_step)
                    self.frames.append(self.env.render(mode='rgb_array'))
                    self.observations.append(ob)

                action = self.model.step(ob)
                ob, reward, done, _ = self.env.step(action)
                ob = ob[None] # extract numpy array from LazyFrames object
                episode_reward += reward

                if record_step:
                    self.actions.append(action)
                    self.episode_rewards.append(episode_reward)
                
                    for feed_op in feed_operations:
                        feed_dict[feed_op] = ob

                    for op_name in self.operations:
                        self.activations[op_name].append(
                            get_activations(
                                session = session,
                                operation_name = self.operations[op_name],
                                feed_dict = feed_dict))
                    
                if total_steps % 100 == 0:
                    logger.info(f'Step {total_steps} in episode {n_run}')

            if stop:
                break

        logger.info(f'Done recording {total_steps} steps with a sample modulo of {sample_modulo} '
                    f'in {round(time.time()-start_time, 1)} seconds')
                    
    def replay(self, start=None, stop=None, step=None, mode='frames'):

        def reduce_observations():
            '''Reduce observation tensors of shape [batch_size, x, y, last_n_frames] to shape [x,y].'''
            # Remove empty dimension and reduce by calculating the mean over the last remaining dimension 
            reduced = [np.add.reduce(np.squeeze(ob), axis=2) / ob.shape[-1] for ob in self.observations]
            return reduced 

        if mode == 'frames':
            tape = self.frames
        elif mode == 'observations':
            tape = reduce_observations()
        for image in tape[start:stop+1:step]:
            print(image.shape)
            plt.imshow(image, cmap='gray')
            plt.show()