import argparse

from train.trainer import Trainer

parser=argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--cfg', type=str, default='configs/default.yaml')
flags=parser.parse_args()

def train():
    trainer=Trainer(flags.cfg)
    trainer.train()

if __name__=="__main__":
    name2func={
        'train':train
    }
    name2func[flags.task]()