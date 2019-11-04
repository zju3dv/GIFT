from easydict import EasyDict
import os

cfg = EasyDict()

cfg.project_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg.data_dir=os.path.join(cfg.project_dir,'data')
cfg.record_dir=os.path.join(cfg.data_dir,'record')
cfg.model_dir=os.path.join(cfg.data_dir,'model')
cfg.cache_dir=os.path.join(cfg.data_dir,'cache')
