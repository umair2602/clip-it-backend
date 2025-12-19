"""
TalkNet Loss Functions
Based on: https://github.com/TaoRuijie/TalkNet-ASD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class lossAV(nn.Module):
    """Audio-Visual loss for active speaker detection."""
    
    def __init__(self):
        super(lossAV, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, output, labels=None):
        """Forward pass.
        
        Args:
            output: Model output logits (B*T, 2)
            labels: Ground truth labels (optional, for training)
            
        Returns:
            If labels provided: (loss, scores, labels, predictions)
            Otherwise: Speaking scores (softmax probabilities for class 1)
        """
        # Get softmax scores
        scores = F.softmax(output, dim=1)
        
        if labels is not None:
            # Training mode
            loss = self.criterion(output, labels)
            predictions = torch.argmax(output, dim=1)
            return loss, scores, labels, predictions
        else:
            # Inference mode - return speaking scores
            # Score for "speaking" class (index 1)
            speaking_scores = scores[:, 1]
            return speaking_scores


class lossA(nn.Module):
    """Audio-only loss."""
    
    def __init__(self):
        super(lossA, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, output, labels=None):
        scores = F.softmax(output, dim=1)
        
        if labels is not None:
            loss = self.criterion(output, labels)
            predictions = torch.argmax(output, dim=1)
            return loss, scores, labels, predictions
        else:
            return scores[:, 1]


class lossV(nn.Module):
    """Visual-only loss."""
    
    def __init__(self):
        super(lossV, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, output, labels=None):
        scores = F.softmax(output, dim=1)
        
        if labels is not None:
            loss = self.criterion(output, labels)
            predictions = torch.argmax(output, dim=1)
            return loss, scores, labels, predictions
        else:
            return scores[:, 1]
