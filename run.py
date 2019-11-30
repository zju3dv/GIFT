import argparse

from dataset.correspondence_database import CorrespondenceDatabase
from dataset.relative_pose_dataset import get_rel_pair_dataset
from train.evaluation import EvaluationWrapper
from train.trainer import Trainer

parser=argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--cfg', type=str, default='configs/GIFT-stage1.yaml')
parser.add_argument('--det_cfg',type=str,default='configs/eval/superpoint_det.yaml')
parser.add_argument('--desc_cfg',type=str,default='configs/eval/gift_pretrain_desc.yaml')
parser.add_argument('--match_cfg',type=str,default='configs/eval/match_v2.yaml')
flags=parser.parse_args()

def train():
    trainer = Trainer(flags.cfg)
    trainer.train()

def eval_original():
    database=CorrespondenceDatabase()
    evaluator=EvaluationWrapper(flags.det_cfg,flags.desc_cfg,flags.match_cfg)
    evaluator.evaluate_homography_original(database.escale_set, 'es')
    evaluator.evaluate_homography_original(database.erotate_set, 'er')
    evaluator.evaluate_homography_original(database.hi_set, 'hi')
    evaluator.evaluate_homography_original(database.hv_set, 'hv')

def eval():
    database=CorrespondenceDatabase()
    evaluator=EvaluationWrapper(flags.det_cfg,flags.desc_cfg,flags.match_cfg)
    evaluator.evaluate_homography(database.escale_set,'es')
    evaluator.evaluate_homography(database.erotate_set,'er')
    evaluator.evaluate_homography(database.hi_set,'hi')
    evaluator.evaluate_homography(database.hv_set,'hv')

def rel_pose():
    sps_100_200_first_100_dataset={
        'dataset':get_rel_pair_dataset(100,200,'st_peters_square_dataset')[:100],
        'name':'sps_100_200_first_100'
    }
    cur_rel_pose_dataset=sps_100_200_first_100_dataset
    dataset=cur_rel_pose_dataset['dataset']
    dataset_name=cur_rel_pose_dataset['name']
    evaluator=EvaluationWrapper(flags.det_cfg,flags.desc_cfg,flags.match_cfg)
    evaluator.relative_pose_estimation(dataset,dataset_name)



if __name__=="__main__":
    name2func={
        'train':train,
        'eval_original':eval_original,
        'eval':eval,
        'rel_pose': rel_pose,
    }
    name2func[flags.task]()