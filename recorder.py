import time

import logging
from importlib import reload
reload(logging)
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(level=logging.INFO) 

import time

import matplotlib.pyplot as plt

def get_activations(session, operation_name, feed_dict):
    # Use the first output tensor of the operation
    tensor = session.graph.get_operation_by_name(operation_name).outputs[0]
    activations = tensor.eval(session=session, 
                              feed_dict=feed_dict)
    return activations

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
        

    def record(self, session, feed_operations, max_episodes=1, max_steps=None):
        '''
        TODO Make feed_operations accept both Tensorflow operations and operation names
        TODO Use Tensorflow writer instead of keeping records in memory
        TODO Think about max_episodes and max_steps order
        '''
        
        logger.info(f'Start recording for {max_steps} steps or {max_episodes} episodes '
                     '(whichever is reached first)')

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
                
                if max_steps:
                    if total_steps >= max_steps:
                        stop = True
                        break

                n_step += 1
                total_steps += 1
                self.n_runs.append(n_run)
                self.n_steps.append(n_step)

                self.frames.append(self.env.render(mode='rgb_array'))
                
                self.observations.append(ob)
                action = self.model(ob)[0]
                self.actions.append(action)

                ob, reward, done, _ = self.env.step(action)
                ob = ob[None] # extract numpy array from LazyFrames object

                episode_reward += reward
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

        logger.info(f'Done recording {total_steps} steps in {round(time.time()-start_time, 1)} seconds')
                    
    def replay(self, start=None, stop=None, step=None):
        for frame in self.frames[start:stop+1:step]:
            plt.imshow(frame)
            plt.show()