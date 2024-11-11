import unittest
import torch
from utils.utils import entropy_loss, recall_at_1

class TestUtils(unittest.TestCase):
    
    def test_entropy(self):
        # Create sample data
        x = torch.randn(10, 128)  # Example tensor with 10 vectors of 128 dimensions
        
        # Compute entropy
        entropy_value = entropy_loss(x)
        
        # Print the entropy value for inspection
        print(f"Entropy: {entropy_value.item()}")
        
        # Assert that the entropy is a finite number
        self.assertTrue(torch.isfinite(entropy_value).item(), "Entropy should be a finite number")
        
        # Optionally, you can check if the entropy is within an expected range
        # For random data, the entropy should be reasonably high
        self.assertGreater(entropy_value.item(), 0, "Entropy should be greater than 0")
        self.assertLess(entropy_value.item(), 10, "Entropy should be less than 10")

    def test_recall_at_1(self):
        # Create sample data
        X = torch.randn(10, 128)  # Example tensor with 10 vectors of 128 dimensions
        labels = torch.randint(0, 2, (10,))  # Example ground truth labels with 2 classes
        
        # Compute recall@1
        recall = recall_at_1(X, labels)
        
        # Print the recall@1 value for inspection
        print(f"Recall@1: {recall}")
        
        # Assert that the recall@1 is a valid probability
        self.assertGreaterEqual(recall, 0, "Recall@1 should be greater than or equal to 0")
        self.assertLessEqual(recall, 1, "Recall@1 should be less than or equal to 1")