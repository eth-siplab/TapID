"""
Constants and enums for the TapID project.

This module defines the core constants and enumerations used throughout the TapID project,
including finger types, sensor configurations, and data processing parameters.
"""

from enum import Enum, auto
from typing import List, Union
import numpy as np

# ============================================================================
# Enums
# ============================================================================

class Fingers(Enum):
    """
    Enumeration of all possible finger types, including left and right hand fingers.
    
    This enum provides a type-safe way to reference finger types throughout the codebase.
    Each finger type is assigned a unique integer value, and the enum provides
    string representations for easy debugging and logging.
    
    The order follows standard anatomical convention:
    - 0: thumb
    - 1: index finger
    - 2: middle finger
    - 3: ring finger
    - 4: little finger (pinky)
    
    Attributes
    ----------
    LEFT_THUMB : int
        Left thumb (0)
    LEFT_INDEX : int
        Left index finger (1)
    LEFT_MIDDLE : int
        Left middle finger (2)
    LEFT_RING : int
        Left ring finger (3)
    LEFT_PINKY : int
        Left pinky finger (4)
    RIGHT_THUMB : int
        Right thumb (5)
    RIGHT_INDEX : int
        Right index finger (6)
    RIGHT_MIDDLE : int
        Right middle finger (7)
    RIGHT_RING : int
        Right ring finger (8)
    RIGHT_PINKY : int
        Right pinky finger (9)
    OOC : int
        Out of class (10)
    """
    LEFT_THUMB = 0
    LEFT_INDEX = 1
    LEFT_MIDDLE = 2
    LEFT_RING = 3
    LEFT_PINKY = 4
    RIGHT_THUMB = 5
    RIGHT_INDEX = 6
    RIGHT_MIDDLE = 7
    RIGHT_RING = 8
    RIGHT_PINKY = 9
    OOC = 10

    @classmethod
    def get_all_fingers(cls) -> List['Fingers']:
        """Get a list of all finger types (excluding OOC)."""
        return [f for f in cls if f != cls.OOC]

    @classmethod
    def get_num_fingers(cls) -> int:
        """Get the total number of finger types (excluding OOC)."""
        return len(cls.get_all_fingers())

class FingerPosition(Enum):
    """
    Enumeration of finger positions without hand distinction.
    
    This enum provides a simplified way to reference finger positions (0-4)
    regardless of which hand they belong to. This is useful for models that
    don't distinguish between left and right hands.
    
    The order follows standard anatomical convention:
    - 0: thumb
    - 1: index finger
    - 2: middle finger
    - 3: ring finger
    - 4: little finger (pinky)
    
    Attributes
    ----------
    THUMB : int
        Thumb position (0)
    INDEX : int
        Index finger position (1)
    MIDDLE : int
        Middle finger position (2)
    RING : int
        Ring finger position (3)
    PINKY : int
        Pinky/little finger position (4)
    """
    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4

    @classmethod
    def get_all_positions(cls) -> List['FingerPosition']:
        """Get a list of all finger positions."""
        return list(cls)

    @classmethod
    def get_num_positions(cls) -> int:
        """Get the total number of finger positions."""
        return len(cls)

class Sensor(Enum):
    """
    Enumeration of all available sensors.
    
    Attributes
    ----------
    SENSOR0 : int
        First sensor (0)
    SENSOR1 : int
        Second sensor (1)
    SENSOR2 : int
        Third sensor (2)
    SENSOR3 : int
        Fourth sensor (3)
    """
    SENSOR0 = 0
    SENSOR1 = 1
    SENSOR2 = 2
    SENSOR3 = 3

    @classmethod
    def get_all_sensors(cls) -> List['Sensor']:
        """Get a list of all available sensors."""
        return list(cls)

    @classmethod
    def get_num_sensors(cls) -> int:
        """Get the total number of available sensors."""
        return len(cls)

class Axis(Enum):
    """
    Enumeration of all available axes.
    
    Attributes
    ----------
    X : int
        X-axis (0)
    Y : int
        Y-axis (1)
    Z : int
        Z-axis (2)
    """
    X = 0
    Y = 1
    Z = 2

    @classmethod
    def get_all_axes(cls) -> List['Axis']:
        """Get a list of all available axes."""
        return list(cls)

# ============================================================================
# Constants
# ============================================================================

# IMU Configuration
IMU_FS = 1266  # IMU sampling frequency in Hz

# ============================================================================
# Utility Functions
# ============================================================================

def map_finger_to_position(finger_idx: Union[int, np.integer]) -> int:
    """
    Map a finger index to its position type (thumb, index, middle, ring, pinky) regardless of hand.
    
    Parameters
    ----------
    finger_idx : Union[int, np.integer]
        Index of the finger (0-10)
        
    Returns
    -------
    int
        Position index (0-4):
        - 0: thumb
        - 1: index finger
        - 2: middle finger
        - 3: ring finger
        - 4: pinky
        
    Examples
    --------
    >>> map_finger_to_position(0)  # Left thumb
    0
    >>> map_finger_to_position(1)  # Left index
    1
    >>> map_finger_to_position(5)  # Right thumb
    0
    >>> map_finger_to_position(10)  # OOC
    10
    """
    if not isinstance(finger_idx, (int, np.integer)):
        raise TypeError(f"finger_idx must be an integer, got {type(finger_idx)}")
    
    if finger_idx < 0 or finger_idx > Fingers.OOC.value:
        raise ValueError(f"finger_idx must be in range [0-10], got {finger_idx}")
    
    # Map to position (0-4) using modulo 5
    # This works for both left and right hand fingers
    return finger_idx % 5