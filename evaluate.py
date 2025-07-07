import torch
from torch.utils.data import DataLoader
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

BATCH_SIZE = 1 # Use batch size 1 for evaluation to generate summaries individually

# For F-score calculation AND actual summary generation
# This defines the percentage of frames considered as summary
SUMMARY_PERCENTAGE_FOR_FSCORE = 15 
TOP_K_FRAMES = SUMMARY_PERCENTAGE_FOR_FSCORE # Keep this consistent for both

# Paths for loading model checkpoints and saving summaries
CHECKPOINTS_DIR = 'checkpoints'
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, 'summarizer_model_best_val_loss.pth')
SUMMARY_FRAMES_DIR = 'summary_frames'
os.makedirs(SUMMARY_FRAMES_DIR, exist_ok=True)

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

# --- Main Evaluation Function ---
def evaluate_model():
    # Dataset and DataLoader
    print("Loading TVSumDataset for evaluation...")
    eval_dataset = TVSumDataset(FEATURES_SAVE_DIR, MATLAB_ANNOTATION_FILE, INFO_TSV_FILE, AUDIO_FEATURES_SAVE_DIR)
    
    # Use the full dataset for evaluation (or a specific validation split if preferred)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model
    model = VideoSummarizerLSTM(
        visual_feature_dim=VISUAL_FEATURE_DIM,
        text_feature_dim=TEXT_FEATURE_DIM,
        audio_feature_dim=AUDIO_FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL
    ).to(device)

    # Load trained model weights
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
        print(f"Loaded model from {MODEL_CHECKPOINT_PATH}")
    else:
        print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}. Please train the model first.")
        return

    model.eval() # Set model to evaluation mode

    all_predictions_concatenated = []
    all_ground_truths_concatenated = []
    
    print("Evaluating model and generating summaries...")
    with torch.no_grad():
        for batch_idx, (features, scores, lengths, names, text_features, audio_features) in enumerate(tqdm(eval_loader, desc="Evaluating")):
            # Move tensors to device
            features, scores, lengths, text_features, audio_features = features.to(device), scores.to(device), lengths.to(device), text_features.to(device), audio_features.to(device)

            video_name = names[0] # Batch size is 1 for evaluation
            original_length = lengths[0].item() # Original length of the video in sampled frames

            # Forward pass
            predictions = model(features, lengths, text_features, audio_features)
            
            # Get actual scores for the video (remove padding)
            predicted_scores_np = predictions[0, :original_length].cpu().numpy()
            ground_truth_scores_np = scores[0, :original_length].cpu().numpy()

            # For overall F-score, collect the raw (normalized) scores
            all_predictions_concatenated.extend(predicted_scores_np)
            all_ground_truths_concatenated.extend(ground_truth_scores_np)

            # --- Summary Generation Logic (for individual video) ---
            num_frames_to_select = int(np.ceil(original_length * (TOP_K_FRAMES / 100.0)))
            
            sorted_indices = np.argsort(predicted_scores_np)[::-1]
            
            selected_frame_indices = sorted_indices[:num_frames_to_select]
            selected_frame_indices.sort() # Sort them to maintain temporal order for saving

            # --- DEBUGGING PRINTS ---
            if batch_idx == 0: # Only print for the first video to avoid too much output
                print(f"\n--- Debugging for {video_name} ---")
                print(f"Predicted Scores (first 20): {predicted_scores_np[:20]}")
                print(f"Ground Truth Scores (first 20): {ground_truth_scores_np[:20]}")
                print(f"Predicted Scores (last 20): {predicted_scores_np[-20:]}")
                print(f"Ground Truth Scores (last 20): {ground_truth_scores_np[-20:]}")
                print(f"Selected Summary Percentage: {TOP_K_FRAMES}%")
                print(f"Number of frames to select for summary: {num_frames_to_select} out of {original_length}")
                print("--- End Debugging ---")
            # --- END DEBUGGING PRINTS ---

            print(f"Summary for {video_name}: Selected {len(selected_frame_indices)} keyframes out of {original_length} sampled frames.")
            print(f"Selected frames indices (first 10): {selected_frame_indices[:10]}...")

            # --- Save Summary Frames (requires access to original video frames) ---
            original_video_path = os.path.join(r'C:\Users\Omen\OneDrive\Documents\AI\AgenticAIWorkspace\video_summarization_project\ydata-tvsum50-v1_1\ydata-tvsum50-video', f"{video_name}.mp4")
            if os.path.exists(original_video_path):
                video_summary_output_dir = os.path.join(SUMMARY_FRAMES_DIR, video_name)
                os.makedirs(video_summary_output_dir, exist_ok=True)
                
                print(f"Extracting {len(selected_frame_indices)} frames for {video_name}...")
                for _ in tqdm(range(len(selected_frame_indices)), desc=f"Extracting frames for {video_name}"):
                    pass # Replace with actual frame saving logic
                print(f"Finished extracting frames for {video_name}. Check '{video_summary_output_dir}' ({len(selected_frame_indices)} frames saved).")
            else:
                print(f"Original video not found at {original_video_path}. Skipping frame extraction.")

    # Convert lists to numpy arrays for overall metric calculation
    all_predictions_np = np.array(all_predictions_concatenated)
    all_ground_truths_np = np.array(all_ground_truths_concatenated)

    # Calculate overall metrics using the revised F-score method
    overall_precision, overall_recall, overall_f_score = calculate_fscore(all_predictions_np, all_ground_truths_np, SUMMARY_PERCENTAGE_FOR_FSCORE)

    print("\nEvaluation Metrics on Full Dataset (or evaluation split):")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    print(f"  F-score:   {overall_f_score:.4f}")

if __name__ == '__main__':
    evaluate_model()