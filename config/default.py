"""
Default configuration for TapID experiments.

This module contains default configuration parameters for running TapID experiments.
These parameters can be overridden by command-line arguments.
"""

# Data parameters
TAPS_PER_BLOCK = 30
DOWNSAMPLING_FACTOR = 1
DOWNSAMPLING_METHOD = 'mean'

# Training parameters
MAX_EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

# Default sessions to use
DEFAULT_SESSIONS = [0, 1, 2, 3, 4, 5]

# Default sensors to use
DEFAULT_SENSORS = [0, 1, 2, 3]

# Default participants to use (None means all)
DEFAULT_PARTICIPANTS = None

# Default fingers to classify
DEFAULT_FINGERS_TO_CLASSIFY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 