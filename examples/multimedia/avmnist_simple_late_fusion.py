import sys
import os
sys.path.append(os.getcwd())
from unimodals.common_models import LeNet, MLP, Constant
import torch
from torch import nn
from datasets.avmnist.get_data import get_dataloader
from fusions.common_fusions import Concat, Average
from training_structures.Supervised_Learning import train, test
# from training_structures.Supervised_Learning_SI import train, test
from pytorch_lightning import seed_everything
import wandb
# try doing it just on the head.fc.weight.grad for now
seed_everything(0)

torch.multiprocessing.set_sharing_strategy('file_system')

wandb.init(project="avmnist", name="late fusion concat lr=0.01") #group="fusion_seeds")

traindata, validdata, testdata = get_dataloader(
    '/mnt/c/Users/Haoli Yin/Documents/MultiBench-enfusion/avmnist')
channels = 6
encoders = [LeNet(1, channels, 3).cuda(), LeNet(1, channels, 5).cuda()]
head = MLP(channels*40, 100, 10).cuda()

fusion = Concat().cuda()

train(encoders, fusion, head, traindata, validdata, 30,
      optimtype=torch.optim.SGD, lr=0.01, weight_decay=0.0001)

print("Testing:")
model = torch.load('best.pt').cuda()
test(model, testdata, no_robust=True)
