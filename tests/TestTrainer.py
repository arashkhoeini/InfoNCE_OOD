import unittest
from configs.init_configs import init_config
import utils.utils as utils
import torch
from pretrainer import Trainer as Pretrainer
from models.moco2 import MoCo
from torchvision import models

class TestTrainer(unittest.TestCase):
    
    def setUp(self) -> None:
        self.resume = ''
        self.configs = init_config('configs/configs.yml', [])
        # create dummy loaders with random values
        num_samples = 100
        num_features = 3 * 32 * 32  # Assuming CIFAR-10 like data
        dummy_images = torch.randn(num_samples, 3, 32, 32)
        dummy_labels = torch.zeros(num_samples, dtype=torch.long)  # All labels are 0 for simplicity
        
        # Create TensorDataset instances
        train_dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
        val_dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
        test_dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
        
        # Create DataLoader instances
        batch_size = self.configs.dataset.batch_size
        self.loaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        }
        self.model = MoCo(models.__dict__[self.configs.model.encoder], dim=self.configs.model.feature_dim, K=self.configs.model.num_negs, T=self.configs.model.temperature)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = Pretrainer(self.model, self.device, self.loaders, self.configs, resume=self.resume)

    def test__filter_logits_based_on_difficulty(self):
        logits = torch.tensor([[ 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
                            [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 
                            [ 1,3, 2, 5, 4, 7, 11, 6, 9, 8, 10]], dtype=torch.float32)
        
        # Test case 1
        difficulty = (0, .3)
        expected_output = torch.tensor([[2., 3., 4.],
                                        [1., 2., 3.],
                                        [2., 3., 4.]])
        results = self.trainer._filter_logits_based_on_difficulty(logits, difficulty)
        self.assertTrue(torch.equal(results, expected_output))

        # Test case 2
        difficulty = (.6, .9)
        expected_output = torch.tensor([[ 8.,  9., 10.],
                                        [ 7.,  8.,  9.],
                                        [ 8.,  9., 10.]])
        results = self.trainer._filter_logits_based_on_difficulty(logits, difficulty)
        self.assertTrue(torch.equal(results, expected_output))

        # Test case 3
        difficulty = (.7, 1)
        expected_output = torch.tensor([[ 9., 10., 11.],
                                        [ 8.,  9., 10.],
                                        [ 9., 10., 11.]])
        results = self.trainer._filter_logits_based_on_difficulty(logits, difficulty)
        self.assertTrue(torch.equal(results, expected_output))

    def tearDown(self) -> None:
        self.trainer = None

