import os
import torch
import numpy as np
import random
import wandb


def set_seed(env, seed):
  np.random.seed(seed)
  random.seed(seed)
  seed_torch(seed)
  env.seed(seed)
  return env


def seed_torch(seed):
  torch.manual_seed(seed)
  if torch.backends.cudnn.enabled:
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True


def setup_dir(folder_name: str):
  # Creating required dir structure
  
  # Data dir
  if not os.path.isdir("data"):
    os.mkdir("data")
  if not os.path.isdir(f"data/{folder_name}"):
    os.mkdir(f"data/{folder_name}")
  
  # Models dir
  if not os.path.isdir("models"):
    os.mkdir("models")
  if not os.path.isdir(f"models/{folder_name}"):
    os.mkdir(f"models/{folder_name}")
  if not os.path.isdir(f"models/{folder_name}/RL"):
    os.mkdir(f"models/{folder_name}/RL")
  if not os.path.isdir(f"models/{folder_name}/BC"):
    os.mkdir(f"models/{folder_name}/BC")


class Queue:
  def __init__(self, n: int = 10):
    self.max_length = n
    self.storage = []
    self.pos = 0
    self.is_max_capacity = lambda: len(self.storage) == self.max_length
    self.min = lambda: min(self.storage)
  
  def add(self, num: float):
    if self.is_max_capacity():
      self.storage[self.pos] = num
      self.pos = (self.pos + 1) % self.max_length
    else:
      self.storage.append(num)
      self.pos = (self.pos + 1) % self.max_length
     
  def get_avg(self):
    return sum(self.storage)/len(self.storage) \
      if self.is_max_capacity() else None
  
  def __len__(self):
    return len(self.storage)


class Stopper:
  def __init__(self, condition: object, reward_thresh: float, n: int = 10, active: bool = True):
    self.size = n
    self.storage = Queue(n)
    self.condition = condition
    self.active = active
    self.reward_thresh = reward_thresh
    
  def update(self, num: float):
    self.storage.add(min(num, self.reward_thresh))

  def check_cond(self):
    # Must update this method when stop condition is changed in RL_main.py
    return self.condition(self.storage.get_avg(), self.storage.min()) if self.active and self.storage.is_max_capacity() else False


class FileNameGen:
  @staticmethod
  def generate_model_name(env_config, env_name):
    env_name = f"{env_name.split('_')[0][0]}{env_name.split('_')[1]}"
    env_settings = f"{env_config['initial_random']}r-{env_config['main_engine_mod']}m-{env_config['side_engine_mod']}s"
    return f"RL-{env_name}-{env_settings}"

  @staticmethod
  def generate_run_name(model_config, env_config):
    env_settings = f"{env_config['initial_random']}r-{env_config['main_engine_mod']}m-{env_config['side_engine_mod']}s"

    # Extracting model parameters from filename
    data = model_config['weights_filename'].split("-")
    r, m, s = data[2:]
    model_settings = f"{r}-{m}-{s}"
    return f"RL model: {model_settings} env: {env_settings}"


class Scheduler:
    class Job:
        def __init__(self, file_name, model_config, env_config):
            self.file_name = file_name
            self.model_config = model_config
            self.env_config = env_config
            self.training = model_config["training"]


    def __init__(self, agent, wandb_config):
        self.agent = agent
        self.default_env_config = {
            "initial_random": self.agent.env.initial_random,
            "main_engine_power": self.agent.env.main_engine_power,
            "side_engine_power": self.agent.env.side_engine_power
        }
        self.env_name = agent.env.name
        self.wandb_config = wandb_config
        self.jobs = []

    def add_batch_jobs(self, model_configs, env_configs):
        model_configs, env_configs = list(model_configs.values()), list(env_configs.values())
        for i in range(len(model_configs)):
            model_config, env_config = model_configs[i], env_configs[i]
            self.jobs.append(self.Job(
                FileNameGen.generate_model_name(env_config, self.env_name),
                model_config,
                env_config
            ))

    def run_jobs(self):
        for job in self.jobs:
          wandb.init(
              project="hri-collab", 
              entity="monash-deep-neuron", 
              config=self.wandb_config,
              name=FileNameGen.generate_run_name(job.model_config, job.env_config) if job.model_config["weights_filename"] is not None else None
          )
          
          self._update_env_config(job)
          if job.training:
              self.agent.train(job.model_config["time_steps"], stopper=job.model_config["stopper"], weights_filename=job.file_name)
          else:
              weights_filename = job.model_config["weights_filename"] if job.model_config["weights_filename"] is not None else job.file_name
              self.agent.evaluate(job.model_config["time_steps"], reward_goal=job.model_config["reward_max"], weights_filename=weights_filename)
          
          wandb.finish(quiet=True)
            

    def _update_env_config(self, job):
        self.agent.env.initial_random = job.env_config["initial_random"] * self.default_env_config["initial_random"]
        self.agent.env.main_engine_power = job.env_config["main_engine_mod"] * self.default_env_config["main_engine_power"]
        self.agent.env.side_engine_power = job.env_config["side_engine_mod"] * self.default_env_config["side_engine_power"]

