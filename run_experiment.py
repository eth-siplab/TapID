#!/usr/bin/env python3
"""
Experiment runner for TapID.

This script runs the experiments for the TapID paper, training models on either 
cross-participant or cross-block data. It implements the experiments described in 
the TapID paper.
"""

import os
import argparse
import pickle
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from skorch import NeuralNet

# Import from the TapID package
from tapid.model import VGGNetmax, get_class_from_probs
from tapid.data_utils import (
    load_data,
    get_data_arrays,
    filter_fingers,
    get_selected_data,
    downsample
)
from tapid.preprocessing import DataNormaliser
from tapid.constants import Sensor, map_finger_to_position, IMU_FS

# Import configuration
from config import (
    TAPS_PER_BLOCK,
    DOWNSAMPLING_FACTOR,
    DOWNSAMPLING_METHOD,
    MAX_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    DEFAULT_SESSIONS,
    DEFAULT_SENSORS,
    DEFAULT_PARTICIPANTS,
    DEFAULT_FINGERS_TO_CLASSIFY
)

def compute_overall_metrics(results):
    """Compute overall metrics from a list of result dictionaries."""
    f1_scores_per_finger = []
    accuracies = []
    
    for result in results:
        # Extract F1 scores for each finger (0-4)
        finger_f1_scores = [result['classification_report'][str(k)]['f1-score'] 
                          for k in range(5) if str(k) in result['classification_report']]
        f1_scores_per_finger.append(finger_f1_scores)
        
        # Extract accuracy
        accuracies.append(result['classification_report']['accuracy'])
    
    # Convert to numpy array for easier computation
    f1_scores_per_finger = np.array(f1_scores_per_finger)
    
    # Calculate mean F1 score across fingers first, then across folds
    mean_f1_per_fold = f1_scores_per_finger.mean(axis=1)
    
    return {
        'f1_mean': np.mean(mean_f1_per_fold),  # Mean across folds
        'f1_std': np.std(mean_f1_per_fold),    # Std across folds
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'f1_scores_per_finger': f1_scores_per_finger
    }

def main():
    parser = argparse.ArgumentParser(description='Run TapID experiments')
    parser.add_argument('--mode', type=str, default='crossparticipant',
                      choices=['crossparticipant', 'crossblock'],
                      help='Training mode (default: crossparticipant)')
    parser.add_argument('--sensors', type=int, nargs='+', default=DEFAULT_SENSORS,
                      help=f'List of sensors to use (default: {DEFAULT_SENSORS})')
    parser.add_argument('--sessions', type=int, nargs='+', default=DEFAULT_SESSIONS,
                      help=f'List of sessions to use (default: {DEFAULT_SESSIONS})')
    parser.add_argument('--participants', type=int, nargs='+', default=DEFAULT_PARTICIPANTS,
                      help=f'List of participants to use (default: {DEFAULT_PARTICIPANTS})')
    parser.add_argument('--fingers', type=int, nargs='+', default=DEFAULT_FINGERS_TO_CLASSIFY,
                      help=f'List of fingers to classify (default: {DEFAULT_FINGERS_TO_CLASSIFY})')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                      help='Name of the experiment for saving results')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory containing the data')
    args = parser.parse_args()

    # Convert sensor numbers to Sensor enum values
    sensors = [Sensor(s) for s in args.sensors]

    # Set up experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.mode}_sensor_{''.join([str(s.value) for s in sensors])}_{len(args.sessions)//2}"

    # Load and prepare data
    all_data, all_labels, all_peaks, all_sessions = load_data(args.data_dir)
    data, labels, participant_ids, sessions = get_data_arrays(
        all_data, all_labels, all_peaks, all_sessions,
        downsampling_factor=DOWNSAMPLING_FACTOR,
        downsampling=DOWNSAMPLING_METHOD
    )
    
    # First use get_selected_data to filter by participants, labels, sensors, and sessions
    data, labels, participant_ids, indices, sessions = get_selected_data(
        data, labels, participant_ids, sessions,
        participants=args.participants,
        labels=args.fingers,
        sensors=[s.value for s in sensors],
        sessions=args.sessions,
        taps_per_block=TAPS_PER_BLOCK
    )
    
    # Then use filter_fingers to map finger indices to position classes
    data, labels, participant_ids, indices, sessions = filter_fingers(
        data, labels, participant_ids, indices, sessions,
        finger_idxs=args.fingers
    )
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    net = NeuralNet(
        VGGNetmax,
        module__nsensor=len(sensors),
        criterion=torch.nn.NLLLoss,
        criterion__reduction='mean',
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        optimizer=torch.optim.Adam,
        iterator_train__shuffle=True,
        iterator_train__drop_last=True,
        train_split=None,
        device=device,
        verbose=True
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'crossparticipant':
        # Set up cross-participant validation
        logo = LeaveOneGroupOut()
        
        # Initialize results storage
        accuracy = []
        results = []
        
        for train_idx, val_idx in logo.split(data, labels, participant_ids):
            print(f"Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")
            
            # Split data
            train_data, train_labels = data[train_idx], labels[train_idx]
            val_data, val_labels = data[val_idx], labels[val_idx]
            
            # Normalize data
            normalizer = DataNormaliser(jerk=True)
            train_data = normalizer.fit_transform(train_data)
            val_data = normalizer.transform(val_data)
            
            # Train model
            net.fit(train_data, train_labels)
            
            # Evaluate model
            val_probs = net.predict_proba(val_data)
            val_pred = get_class_from_probs(val_probs)
            
            # Calculate metrics
            conf_matrix = confusion_matrix(val_labels, val_pred)
            class_report_dict = classification_report(val_labels, val_pred, output_dict=True)
            acc = accuracy_score(val_labels, val_pred)
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(val_labels, val_pred))
            
            # Create output dictionary
            output_dict = {
                'confusion_matrix': conf_matrix,
                'classification_report': class_report_dict,
                'true_labels': val_labels,
                'predicted_labels': val_pred,
                'probabilities': val_probs
            }
            
            results.append(output_dict)
            accuracy.append(acc)
            
            print(f"Validation accuracy: {acc:.4f}")
            print("="*100)
            
            # Clear GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Compute overall metrics
        overall_metrics = compute_overall_metrics(results)
        
        print(f"Mean accuracy: {overall_metrics['accuracy_mean']:.4f}")
        print(f"Accuracy standard deviation: {overall_metrics['accuracy_std']:.4f}")
        print(f"Mean F1 score (across fingers): {overall_metrics['f1_mean']:.4f}")
        print(f"F1 score standard deviation: {overall_metrics['f1_std']:.4f}")
        
        # Print per-finger F1 scores
        mean_f1_per_finger = overall_metrics['f1_scores_per_finger'].mean(axis=0)
        std_f1_per_finger = overall_metrics['f1_scores_per_finger'].std(axis=0)
        for finger, (mean_f1, std_f1) in enumerate(zip(mean_f1_per_finger, std_f1_per_finger)):
            print(f"Finger {finger} - Mean F1: {mean_f1:.4f}, Std: {std_f1:.4f}")
        
        # Save results
        results_path = os.path.join(args.output_dir, f"{args.experiment_name}_results.pkl")
        
        with open(results_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'accuracy': accuracy,
                'overall_metrics': overall_metrics
            }, f)
        
        print(f"Results saved to {results_path}")
    
    elif args.mode == 'crossblock':
        # Set up cross-block validation
        logo = LeaveOneGroupOut()
        
        # Initialize results storage
        cross_block_experiments = {}
        
        # Perform cross-block experiments
        for participant in np.unique(participant_ids):
            print(f"\nProcessing participant {participant}")
            
            # Filter data for current participant
            participant_mask = participant_ids == participant
            participant_data = data[participant_mask]
            participant_labels = labels[participant_mask]
            participant_sessions = sessions[participant_mask]
            
            # Initialize results storage for this participant
            p_accuracy = []
            p_results = []
            
            # Split sessions into blocks
            session_blocks = np.split(np.array(DEFAULT_SESSIONS), len(DEFAULT_SESSIONS)//2)
            
            for val_sessions in session_blocks:
                print(f"Validating on sessions {val_sessions}. Training on rest.")
                
                # Filter data for current sessions
                val_mask = np.isin(participant_sessions, val_sessions)
                train_mask = ~val_mask

                train_data = participant_data[train_mask]
                train_labels = participant_labels[train_mask]
                val_data = participant_data[val_mask]
                val_labels = participant_labels[val_mask]
                
                # Set up normalizer
                normalizer = DataNormaliser(jerk=True)
                train_data = normalizer.fit_transform(train_data)
                val_data = normalizer.transform(val_data)
                
                # Train model
                net.fit(train_data, train_labels)
                
                # Evaluate model
                val_probs = net.predict_proba(val_data)
                val_pred = get_class_from_probs(val_probs)
                
                # Calculate metrics
                conf_matrix = confusion_matrix(val_labels, val_pred)
                class_report_dict = classification_report(val_labels, val_pred, output_dict=True)
                acc = accuracy_score(val_labels, val_pred)
                
                # Print classification report
                print("\nClassification Report:")
                print(classification_report(val_labels, val_pred))
                
                # Create output dictionary
                output_dict = {
                    'confusion_matrix': conf_matrix,
                    'classification_report': class_report_dict,
                    'true_labels': val_labels,
                    'predicted_labels': val_pred,
                    'probabilities': val_probs
                }
                
                p_results.append(output_dict)
                p_accuracy.append(acc)
                
                print(f"Validation accuracy: {acc:.4f}")
                print("="*100)
                
                # Clear GPU memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Compute overall metrics for this participant
            participant_metrics = compute_overall_metrics(p_results)
            
            # Store results for this participant
            cross_block_experiments[participant] = {
                'results': p_results,
                'accuracy': p_accuracy,
                'overall_metrics': participant_metrics
            }
            
            print(f"Participant {participant} mean accuracy: {participant_metrics['accuracy_mean']:.4f}")
            print(f"Participant {participant} accuracy standard deviation: {participant_metrics['accuracy_std']:.4f}")
            print(f"Participant {participant} mean F1 score (across fingers): {participant_metrics['f1_mean']:.4f}")
            print(f"Participant {participant} F1 score standard deviation: {participant_metrics['f1_std']:.4f}")
            
            # Print per-finger F1 scores for this participant
            mean_f1_per_finger = participant_metrics['f1_scores_per_finger'].mean(axis=0)
            std_f1_per_finger = participant_metrics['f1_scores_per_finger'].std(axis=0)
            for finger, (mean_f1, std_f1) in enumerate(zip(mean_f1_per_finger, std_f1_per_finger)):
                print(f"Participant {participant} - Finger {finger} - Mean F1: {mean_f1:.4f}, Std: {std_f1:.4f}")
            print("="*100)
            print("="*100)
        
        # Save results
        results_path = os.path.join(args.output_dir, f"{args.experiment_name}_results.pkl")
        
        with open(results_path, 'wb') as f:
            pickle.dump(cross_block_experiments, f)
        
        print(f"Results saved to {results_path}")
        
        # Calculate overall metrics across all participants
        all_f1_means = [exp['overall_metrics']['f1_mean'] for exp in cross_block_experiments.values()]
        all_accuracies = [exp['overall_metrics']['accuracy_mean'] for exp in cross_block_experiments.values()]
        
        print(f"Overall mean accuracy: {np.mean(all_accuracies):.4f}")
        print(f"Overall accuracy standard deviation: {np.std(all_accuracies):.4f}")
        print(f"Overall mean F1 score: {np.mean(all_f1_means):.4f}")
        print(f"Overall F1 score standard deviation: {np.std(all_f1_means):.4f}")
        
        # Calculate and print overall per-finger F1 scores
        all_f1_scores = np.array([exp['overall_metrics']['f1_scores_per_finger'] 
                                for exp in cross_block_experiments.values()])
        overall_mean_f1_per_finger = all_f1_scores.mean(axis=(0, 1))  # Average across participants and folds
        overall_std_f1_per_finger = all_f1_scores.std(axis=(0, 1))
        
        print("\nOverall per-finger F1 scores:")
        for finger, (mean_f1, std_f1) in enumerate(zip(overall_mean_f1_per_finger, overall_std_f1_per_finger)):
            print(f"Finger {finger} - Mean F1: {mean_f1:.4f}, Std: {std_f1:.4f}")

if __name__ == "__main__":
    main() 