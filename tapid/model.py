"""
Neural network architectures for TapID.

This module provides neural network architectures for finger classification using
inertial sensor data from wrist-worn devices. It includes the VGGNetmax architecture
used in the TapID paper and utility functions for handling model outputs.
"""

import numpy as np
import torch.nn as nn

def get_class_from_probs(probs):
    """
    Convert probability distributions to class predictions.
    
    This function takes a probability distribution over classes and returns the
    index of the class with the highest probability.
    
    Parameters
    ----------
    probs : numpy.ndarray
        Probability distributions with shape (n_samples, n_classes).
        
    Returns
    -------
    numpy.ndarray
        Class predictions with shape (n_samples,).
    """
    return np.argmax(probs, axis=1)

class VGGNetmax(nn.Module):
    """
    VGG-style neural network for finger classification.
    
    This network architecture is used in the TapID paper for finger classification
    using inertial sensor data. It consists of multiple convolutional layers with
    batch normalization and LeakyReLU activation, followed by max pooling and
    adaptive average pooling.
    
    Parameters
    ----------
    nsensor : int, optional
        Number of sensors used as input to the network. Default is 4.
    nhidden : int, optional
        Number of channels in the first hidden layer. Default is 32.
    nout : int, optional
        Number of output classes. Default is 5.
    """
    def __init__(self, nsensor = 4, nhidden = 32, nout=5):
        """
        Initialize the VGGNetmax network.
        
        Parameters
        ----------
        nsensor : int, optional
            Number of sensors used as input to the network. Default is 4.
        nhidden : int, optional
            Number of channels in the first hidden layer. Default is 32.
        nout : int, optional
            Number of output classes. Default is 5.
        """
        super(VGGNetmax, self).__init__()
        self.net = nn.Sequential(
            # 12 x 128
            nn.Conv1d(3*nsensor, nhidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(nhidden),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(nhidden, nhidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(nhidden),
            nn.LeakyReLU(0.2, inplace=False),
            nn.MaxPool1d(2,2),

            # 32 x 64
            nn.Conv1d(nhidden, nhidden*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(nhidden*2),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(nhidden*2, nhidden*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(nhidden*2),
            nn.LeakyReLU(0.2, inplace=False),
            nn.MaxPool1d(2,2),

            # 64 x 32
            nn.Conv1d(nhidden*2, nhidden*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(nhidden*4),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(nhidden*4, nhidden*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(nhidden*4),
            nn.LeakyReLU(0.2, inplace=False),
            nn.MaxPool1d(2,2),

            # 128 x 16
            nn.Conv1d(nhidden*4, nhidden*8, kernel_size=3, padding=1),
            nn.BatchNorm1d(nhidden*8),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(nhidden*8, nhidden*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(nhidden*8),
            nn.LeakyReLU(0.2, inplace=False),
            nn.MaxPool1d(2,2),

            # 256 x 8
            nn.Conv1d(nhidden*8, nhidden*16, kernel_size=3, padding=1),
            nn.BatchNorm1d(nhidden*16),
            nn.LeakyReLU(0.2, inplace=False),

            nn.Conv1d(nhidden*16, nhidden*16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(nhidden*16),
            nn.LeakyReLU(0.2, inplace=False),
            nn.MaxPool1d(2,2),

            # 512 x 4
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(nhidden*16, nhidden*32, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.fc1 = nn.Linear(nhidden*32, nout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_last_layer = False):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, 3*nsensor, sequence_length).
        return_last_layer : bool, optional
            Whether to return the last layer activations. Default is False.
            
        Returns
        -------
        torch.Tensor
            If return_last_layer is False, returns the softmax probabilities.
            If return_last_layer is True, returns a tuple of (softmax probabilities, last layer activations).
        """
        batch_size = x.size(0)
        x = self.net(x).view(batch_size, -1)
        if return_last_layer:
            l = x.detach().clone()
        x = self.fc1(x)
        if return_last_layer:
            return self.softmax(x), l
        else:
            return self.softmax(x)
