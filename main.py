import torch 
from torch.utils.data import Dataset
import torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

def get_config():

    C = CN()
    C = "update"
