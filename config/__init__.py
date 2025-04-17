"""
TapID Configuration Module

This module provides configuration settings for the TapID experiments.
Import these settings directly from the config package.
"""

from config.default import (
    # Data processing parameters
    TAPS_PER_BLOCK,
    DOWNSAMPLING_FACTOR,
    DOWNSAMPLING_METHOD,
    
    # Model training parameters
    MAX_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    
    # Default experiment settings
    DEFAULT_SESSIONS,
    DEFAULT_SENSORS,
    DEFAULT_PARTICIPANTS,
    DEFAULT_FINGERS_TO_CLASSIFY
)

__all__ = [
    # Data processing parameters
    'TAPS_PER_BLOCK',
    'DOWNSAMPLING_FACTOR',
    'DOWNSAMPLING_METHOD',
    
    # Model training parameters
    'MAX_EPOCHS',
    'BATCH_SIZE',
    'LEARNING_RATE',
    
    # Default experiment settings
    'DEFAULT_SESSIONS',
    'DEFAULT_SENSORS',
    'DEFAULT_PARTICIPANTS',
    'DEFAULT_FINGERS_TO_CLASSIFY'
]

# You can add more configuration files here and import them as needed
# For example:
# from config.custom import * 