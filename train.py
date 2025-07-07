import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np

# Import from your modules
from dataset import TVSumDataset, collate_fn, TEXT_FEATURE_DIM, AUDIO_FEATURE_DIM
from model import VideoSummarizerLSTM

# --- Configuration ---
FEATURES_SAVE_DIR = 'notebooks/data/extracted_features'
AUDIO_FEATURES_SAVE_DIR = 'notebooks/data/extracted_audio_features' # Path to extracted audio features
MATLAB_ANNOTATION_FILE = r'C:\Users\Omen\OneDrive\Documents\AI\AgenticAIWorkspace\video_summarization_project\ydata-tvsum50-v1_1\ydata-tvsum50-matlab\ydata-tvsum50.mat'
INFO_TSV_FILE = r'C:\Users\Omen\OneDrive\Documents\AI\AgenticAIWorkspace\video_summarization_project\ydata-tvsum50-v1_1\ydata-tvsum50-info.tsv'

VISUAL_FEATURE_DIM = 2048 # ResNet-50 features
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50 # You might need more epochs for better performance
GRAD_CLIP_NORM = 1.0 # Gradient clipping to prevent exploding gradients

# For F-score calculation, defines the percentage of frames considered as summary
# This should ideally be consistent with typical summary lengths (e.g., 15% or 20%)
SUMMARY_PERCENTAGE_FOR_FSCORE = 15 

# Paths for saving model checkpoints
CHECKPOINTS_DIR = 'checkpoints'
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, 'summarizer_model_best_val_loss.pth')

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Helper for F-score Calculation (Revised) ---
def calculate_fscore(predictions, ground_truth, summary_percentage=15):
    # Ensure predictions and ground_truth are numpy arrays
    predictions = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else np.array(predictions)
    ground_truth = ground_truth.detach().cpu().numpy() if isinstance(ground_truth, torch.Tensor) else np.array(ground_truth)

    num_frames = len(predictions)
    if num_frames == 0:
        return 0.0, 0.0, 0.0 # Handle empty sequences

    # Determine number of frames to select for summary
    num_frames_to_select = int(np.ceil(num_frames * (summary_percentage / 100.0)))
    if num_frames_to_select == 0: # Ensure at least one frame if sequence is not empty
        num_frames_to_select = 1 if num_frames > 0 else 0

    # Binarize predictions: Select top N frames by predicted score
    binary_predictions = np.zeros_like(predictions, dtype=int)
    if num_frames_to_select > 0:
        # Get indices of frames sorted by predicted score in descending order
        top_pred_indices = np.argsort(predictions)[::-1][:num_frames_to_select]
        binary_predictions[top_pred_indices] = 1
    
    # Binarize ground truth: Select top N frames by ground truth score
    binary_ground_truth = np.zeros_like(ground_truth, dtype=int)
    if num_frames_to_select > 0:
        # Get indices of frames sorted by ground truth score in descending order
        top_gt_indices = np.argsort(ground_truth)[::-1][:num_frames_to_select]
        binary_ground_truth[top_gt_indices] = 1

    # Calculate true positives, false positives, false negatives
    true_positives = np.sum(binary_predictions * binary_ground_truth)
    false_positives = np.sum(binary_predictions * (1 - binary_ground_truth))
    false_negatives = np.sum((1 - binary_predictions) * binary_ground_truth)

    # Calculate Precision, Recall, F-score
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f_score = (2 * precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f_score

# --- Training Function ---
def train_model():
    # Dataset and DataLoader
    print("Loading TVSumDataset...")
    full_dataset = TVSumDataset(FEATURES_SAVE_DIR, MATLAB_ANNOTATION_FILE, INFO_TSV_FILE, AUDIO_FEATURES_SAVE_DIR)
    
    # Split dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model, Optimizer, Loss Function
    model = VideoSummarizerLSTM(
        visual_feature_dim=VISUAL_FEATURE_DIM,
        text_feature_dim=TEXT_FEATURE_DIM,
        audio_feature_dim=AUDIO_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Using MSE for importance score prediction

    best_val_loss = float('inf')

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        train_f_scores = []

        for batch_idx, (features, scores, lengths, names, text_features, audio_features) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)")):
            features, scores, lengths, text_features, audio_features = features.to(device), scores.to(device), lengths.to(device), text_features.to(device), audio_features.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(features, lengths, text_features, audio_features)
            
            # Mask padded parts of the loss calculation
            mask = torch.arange(scores.size(1)).unsqueeze(0).to(device) < lengths.unsqueeze(1)
            loss = criterion(predictions * mask, scores * mask) # Only calculate loss on non-padded elements

            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            optimizer.step()
            total_loss += loss.item()

            # Calculate F-score for training batch (on non-padded parts)
            for i in range(features.size(0)):
                seq_len = lengths[i].item()
                gt_scores_seq = scores[i, :seq_len]
                pred_scores_seq = predictions[i, :seq_len]
                # --- Pass summary_percentage for F-score calculation ---
                _, _, f_score = calculate_fscore(pred_scores_seq, gt_scores_seq, SUMMARY_PERCENTAGE_FOR_FSCORE)
                train_f_scores.append(f_score)

        avg_train_loss = total_loss / len(train_loader)
        avg_train_f_score = np.mean(train_f_scores) if train_f_scores else 0

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_f_scores = []
        with torch.no_grad():
            for batch_idx, (features, scores, lengths, names, text_features, audio_features) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Validation)")):
                features, scores, lengths, text_features, audio_features = features.to(device), scores.to(device), lengths.to(device), text_features.to(device), audio_features.to(device)

                predictions = model(features, lengths, text_features, audio_features)
                
                mask = torch.arange(scores.size(1)).unsqueeze(0).to(device) < lengths.unsqueeze(1)
                loss = criterion(predictions * mask, scores * mask)
                val_loss += loss.item()

                for i in range(features.size(0)):
                    seq_len = lengths[i].item()
                    gt_scores_seq = scores[i, :seq_len]
                    pred_scores_seq = predictions[i, :seq_len]
                    # --- Pass summary_percentage for F-score calculation ---
                    _, _, f_score = calculate_fscore(pred_scores_seq, gt_scores_seq, SUMMARY_PERCENTAGE_FOR_FSCORE)
                    val_f_scores.append(f_score)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_f_score = np.mean(val_f_scores) if val_f_scores else 0

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train F-score: {avg_train_f_score:.4f} | Val Loss: {avg_val_loss:.4f} | Val F-score: {avg_val_f_score:.4f}")

        # Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
            print(f"Model saved to {MODEL_CHECKPOINT_PATH} with improved validation loss: {best_val_loss:.4f}")

    print("Training complete.")

if __name__ == '__main__':
    train_model()