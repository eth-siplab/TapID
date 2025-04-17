"""
Data loading and preprocessing utilities for the TapID project.

This module provides functions for loading, preprocessing, and visualizing
finger tap data from inertial sensors. It includes utilities for:
- Loading data from binary files
- Extracting data arrays and windows
- Visualizing data using PCA and UMAP
"""

# ============================================================================
# Imports
# ============================================================================

import os
import pickle
import itertools
from typing import List, Tuple, Optional
import glob

import numpy as np
import logging

from tapid.constants import map_finger_to_position, IMU_FS

# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_folder: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Load data from .npz files in the specified folder.
    
    Args:
        data_folder (str): Directory containing the .npz files
        
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
            - List of data arrays, each with shape (4, 3, time_series)
            - List of label arrays
            - List of peak arrays
            - List of session arrays
    """


    all_data, all_labels, all_peaks, all_sessions = [], [], [], []

    for filename in sorted(os.listdir(data_folder)):
        if filename.endswith('.npz'):
            npz_path = os.path.join(data_folder, filename)
            
            # Load all data from a single .npz file
            with np.load(npz_path) as npz:
                data = npz['data']
                labels = npz['labels']
                peaks = npz['peaks']
                sessions = npz['sessions']

                all_data.append(data)
                all_labels.append(labels)
                all_peaks.append(peaks)
                all_sessions.append(sessions)

    return all_data, all_labels, all_peaks, all_sessions

# ============================================================================
# Data Filtering and Extraction
# ============================================================================

def filter_fingers(data: np.ndarray, labels: np.ndarray, participant_ids: np.ndarray,
                   indices: np.ndarray, sessions: np.ndarray, finger_idxs: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter data to include only specified finger types and map them to position classes.
    
    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_samples, n_sensors, window_size)
    labels : np.ndarray
        Array of shape (n_samples,) containing finger indices
    participant_ids : np.ndarray
        Array of shape (n_samples,) containing participant IDs
    indices : np.ndarray
        Array of shape (n_samples,) containing original indices
    sessions : np.ndarray
        Array of shape (n_samples,) containing session IDs
    finger_idxs : List[int]
        List of finger indices to include
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (filtered_data, mapped_labels, filtered_participant_ids, filtered_indices, filtered_sessions)
        - filtered_data: Data array for selected fingers
        - mapped_labels: Labels mapped to finger positions (0-4)
        - filtered_participant_ids: Participant IDs for selected fingers
        - filtered_indices: Original indices for selected fingers
        - filtered_sessions: Session IDs for selected fingers
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or if finger_idxs is empty
    """
    if not (len(data) == len(labels) == len(participant_ids) == len(indices) == len(sessions)):
        raise ValueError("Input arrays must have the same length")
        
    if len(finger_idxs) == 0:
        raise ValueError("finger_idxs must not be empty")
        
    mask = np.isin(labels, finger_idxs)
    if not np.any(mask):
        raise ValueError("No data found for the specified finger indices")
        
    return (
        data[mask],
        np.fromiter(map(map_finger_to_position, labels[mask]), dtype=int),
        participant_ids[mask],
        indices[mask],
        sessions[mask],
    )

def get_data_arrays(all_data: List[np.ndarray], all_labels: List[np.ndarray], 
                    all_peaks: List[np.ndarray], all_sessions: List[np.ndarray], 
                    window_size: int = 128, downsampling_factor: int = 1, 
                    downsampling: str = "mean"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data arrays from raw data and peaks.
    
    Parameters
    ----------
    all_data : List[np.ndarray]
        List of data arrays, one per participant, each with shape (4, 3, time_series)
    all_labels : List[np.ndarray]
        List of label arrays, one per participant
    all_peaks : List[np.ndarray]
        List of peak indices, one per participant
    all_sessions : List[np.ndarray]
        List of session arrays, one per participant
    window_size : int, optional
        Size of the window around each peak, by default 128
    downsampling_factor : int, optional
        Factor by which to downsample the data, by default 1
    downsampling : str, optional
        Method to use for downsampling ('mean' or 'max'), by default "mean"
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (data, labels, participant_ids, sessions)
        - data: Array of shape (n_samples, 4, 3, window_size)
        - labels: Array of shape (n_samples,)
        - participant_ids: Array of shape (n_samples,)
        - sessions: Array of shape (n_samples,)
        
    Raises
    ------
    ValueError
        If input lists have different lengths or if downsampling method is invalid
    """
    if not (len(all_data) == len(all_labels) == len(all_peaks) == len(all_sessions)):
        raise ValueError("Input lists must have the same length")
        
    if downsampling not in ["mean", "max"]:
        raise ValueError("Downsampling method must be 'mean' or 'max'")
        
    windowed_data, windowed_labels, windowed_participant_ids, windowed_sessions = [], [], [], []
    
    # Calculate half window size, accounting for downsampling
    half_window = (window_size * downsampling_factor) // 2

    for participant_id, participant_data_array in enumerate(all_data):
        participant_labels = all_labels[participant_id]
        peaks = all_peaks[participant_id]
        participant_sessions = all_sessions[participant_id]
        
        # Process each peak in the data
        for i, peak in enumerate(peaks):
            # Calculate window boundaries, accounting for downsampling
            start = max(0, peak - half_window)
            end = min(participant_data_array.shape[2], peak + half_window)
            
            # Extract window from the time dimension (last dimension)
            window = participant_data_array[:, :, start:end]
            
            # Apply downsampling if needed
            if downsampling_factor > 1:
                if downsampling == "mean":
                    window = downsample(window, downsampling_factor, "mean")
                else:  # max
                    window = downsample(window, downsampling_factor, "max")
            
            windowed_data.append(window.reshape(-1, window.shape[-1]))
            windowed_labels.append(participant_labels[i])
            windowed_participant_ids.append(participant_id)
            windowed_sessions.append(participant_sessions[i])

    return (
        np.array(windowed_data),
        np.array(windowed_labels),
        np.array(windowed_participant_ids),
        np.array(windowed_sessions)
    )

def get_selected_data(all_data: np.ndarray, all_labels: np.ndarray, all_participant_ids: np.ndarray, 
                      all_sessions: np.ndarray, participants: List[int], labels: List[int], 
                      sensors: List[int], sessions: List[int], 
                      taps_per_block: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter data to include only specified participants, labels, sensors, and sessions.
    
    Parameters
    ----------
    all_data : np.ndarray
        Array of shape (n_samples, n_sensors, window_size)
    all_labels : np.ndarray
        Array of shape (n_samples,)
    all_participant_ids : np.ndarray
        Array of shape (n_samples,)
    all_sessions : np.ndarray
        Array of shape (n_samples,)
    participants : List[int]
        List of participant IDs to include
    labels : List[int]
        List of labels to include
    sensors : List[int]
        List of sensors to include
    sessions : List[int]
        List of sessions to include
    taps_per_block : Optional[int], optional
        Maximum number of taps to include per block, by default None
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (data, labels, participant_ids, indices, sessions)
        - data: Filtered data array
        - labels: Filtered labels array
        - participant_ids: Filtered participant IDs array
        - indices: Original indices of selected samples
        - sessions: Filtered sessions array
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or if invalid parameters are provided
    """
    if not (len(all_data) == len(all_labels) == len(all_participant_ids) == len(all_sessions)):
        raise ValueError("Input arrays must have the same length")

    if participants is None:
        participants = np.sort(np.unique(all_participant_ids))
        
    if not all(isinstance(x, (list, np.ndarray)) for x in [participants, labels, sensors, sessions]):
        raise ValueError("participants, labels, sensors, and sessions must be lists or numpy arrays")
        
    mask_labels = np.isin(all_labels, labels)
    mask_participants = np.isin(all_participant_ids, participants)
    mask_sessions = np.isin(all_sessions, sessions)
    mask = np.logical_and(np.logical_and(mask_labels, mask_participants), mask_sessions)

    if taps_per_block is not None:
        for p_id in participants:
            for s, l in itertools.product(sessions, labels):
                mask_tmp = (all_sessions == s) & (all_labels == l) & (all_participant_ids == p_id)
                if np.sum(mask_tmp) > taps_per_block:
                    idxs_to_remove = np.where(mask_tmp)[0][taps_per_block:]
                    mask[idxs_to_remove] = False

    indices = np.arange(all_labels.shape[0])[mask]
    return all_data[mask], all_labels[mask], all_participant_ids[mask], indices, all_sessions[mask]

# ============================================================================
# Downsampling
# ============================================================================

def downsample(data: np.ndarray, factor: int, method: str = 'mean') -> np.ndarray:
    """
    Downsample data by a factor along the time dimension.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (4, 3, time_series)
    factor : int
        Factor by which to downsample
    method : str, optional
        Method to use for downsampling ('mean' or 'max'), by default 'mean'
        
    Returns
    -------
    np.ndarray
        Downsampled data array of shape (4, 3, time_series//factor)
        
    Raises
    ------
    ValueError
        If factor is less than 1 or method is invalid
    """
    if factor <= 1:
        return data
        
    if method not in ['mean', 'max']:
        raise ValueError(f"Unknown downsampling method: {method}")
    
    # Get the shape of the input data
    n_sensors, n_axes, time_series = data.shape
    
    # Calculate the new time dimension size
    new_time_size = time_series // factor
    
    # Reshape the data to group samples for downsampling
    # The last dimension is split into groups of size 'factor'
    reshaped = data[:, :, :new_time_size*factor].reshape(n_sensors, n_axes, new_time_size, factor)
    
    # Apply the downsampling method along the last dimension
    if method == 'mean':
        return np.mean(reshaped, axis=-1)
    else:  # max
        return np.max(reshaped, axis=-1)