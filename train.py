import yaml
import os
from trainer.train import train

config = yaml.load(open('config.yml'))
os.environ["CUDA_VISIBLE_DEVICES"] = str(config[config['task']]['gpu'])
train(config)