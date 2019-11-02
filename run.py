import argparse

from dataset.correspondence_database import CorrespondenceDatabase
from train.trainer import Trainer

parser=argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--cfg', type=str, default='configs/default.yaml')
parser.add_argument('--dataset', type=str, default='hi')
parser.add_argument('--kpts',type=str,default='superpoint')
flags=parser.parse_args()

trainer = Trainer(flags.cfg)

def train():
    trainer.train()

def eval():
    dataset=CorrespondenceDatabase().__getattr__(flags.dataset+'_set')
    trainer.evaluate(dataset,flags.dataset+'_set',False,flags.kpts)

if __name__=="__main__":
    name2func={
        'train':train,
        'eval':eval,
    }
    name2func[flags.task]()