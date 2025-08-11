import mne
import numpy as np

def epoch_rejection(raw, events, tmin, tmax, 
                               dropped_len_min, dropped_len_max, 
                               initial_rejection_criteria, 
                               max_iterations=50, 
                               step_size=0.5e-6):
    
    # Create a copy of the initial rejection criteria to modify
    rejection_criteria = initial_rejection_criteria.copy()
    
    
    # Initial attempt
    epochs = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, 
                        reject=rejection_criteria, preload=True)
    
    # Track iterations
    for iteration in range(max_iterations):
        # Count dropped epochs
        dropped_epochs = len(events) - len(epochs)
        
        # Check if we're within the desired range
        if dropped_len_min <= dropped_epochs <= dropped_len_max:
            print(f"Optimal rejection found in {iteration+1} iterations.")
            print(f"Current rejection criteria: {rejection_criteria}")
            print(f"Dropped epochs: {dropped_epochs}")
            return epochs, rejection_criteria
        
        # Adjust threshold based on current dropped epochs
        if dropped_epochs < dropped_len_min:
            # Not enough epochs dropped, lower the threshold
            rejection_criteria['eeg'] -= step_size
        else:
            # Too many epochs dropped, raise the threshold
            rejection_criteria['eeg'] += step_size
        
        # Retry epoching with new criteria
        epochs = mne.Epochs(raw, events=events, tmin=tmin, tmax=tmax, 
                            reject=rejection_criteria, preload=True)
    
    # If max iterations reached without finding optimal rejection
    print("Warning: Could not find optimal rejection criteria within max iterations.")
    return epochs, rejection_criteria
