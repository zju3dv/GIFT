from torch.utils.data import DataLoader

from dataset.correspondence_dataset import CorrespondenceDatabase, CorrespondenceDataset, worker_init_fn


class Trainer:
    def __init__(self):
        pass

    def prepare_training(self):
        pass

    def _init_dataset(self):
        database = CorrespondenceDatabase()
        self.database = database

        train_set = []
        for name in self.config['trainset']:
            train_set += database.__getattribute__(name + "_set")
        self.train_set = CorrespondenceDataset(self.dataset_config, train_set,
                                               min_length=self.config['epoch_steps'] * self.config['batch_size'],
                                               base_scale=self.config['scale_base_ratio'],
                                               base_rotate=self.config['rotate_base_interval'])
        self.train_set = DataLoader(self.train_set, self.config['batch_size'], True, num_workers=self.config['num_workers'], worker_init_fn=worker_init_fn)