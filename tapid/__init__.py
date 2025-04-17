"""
TapID - Rapid Touch Interaction in Virtual Reality using Wearable Sensing

The TapID system uses inertial sensors embedded in wristbands to detect surface
touch events and identify which finger was used for the touch. This enables
natural touch interaction in virtual reality environments, complementing
headset-tracked hand poses.

Main components:
- model.py: Neural network architectures for finger classification
- data_utils.py: Data loading and preprocessing utilities
- constants.py: Constants and enums used throughout the package
- preprocessing.py: Data normalization and preprocessing functions
"""

# Import key components for easier access
from tapid.model import VGGNetmax, get_class_from_probs
from tapid.data_utils import (
    load_data,
    get_data_arrays,
    filter_fingers,
    get_selected_data,
    downsample,
)
from tapid.constants import Sensor, map_finger_to_position, IMU_FS
from tapid.preprocessing import DataNormaliser

__version__ = '1.0.0'

__all__ = [
    # Models
    'VGGNetmax',
    'get_class_from_probs',
    
    # Data utilities
    'load_data',
    'get_data_arrays',
    'filter_fingers',
    'get_selected_data',
    'downsample',
    'DataNormaliser',
    
    # Constants
    'Sensor',
    'map_finger_to_position',
    'IMU_FS',
    
    # Version
    '__version__'
]