import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re
import pandas as pd

# Import SentenceTransformer for text embeddings
from sentence_transformers import SentenceTransformer

# --- Configuration (match feature_extraction.py and audio_feature_extraction.py) ---
TVPARM_BASE_DIR = r'C:\Users\Omen\OneDrive\Documents\AI\AgenticAIWorkspace\video_summarization_project\ydata-tvsum50-v1_1'
MATLAB_ANNOTATION_FILE = os.path.join(TVPARM_BASE_DIR, 'ydata-tvsum50-matlab', 'ydata-tvsum50.mat')
FEATURES_SAVE_DIR = 'notebooks/data/extracted_features' # Where you saved visual features
AUDIO_FEATURES_SAVE_DIR = 'notebooks/data/extracted_audio_features' # Where you saved audio features

# Path to the ydata-tvsum50-info.tsv file
INFO_TSV_FILE = r'C:\Users\Omen\OneDrive\Documents\AI\AgenticAIWorkspace\video_summarization_project\ydata-tvsum50-v1_1\ydata-tvsum50-info.tsv'

# --- Define TEXT_FEATURE_DIM based on the chosen SentenceTransformer model ---
# 'all-MiniLM-L6-v2' outputs 384-dimensional embeddings
TEXT_FEATURE_DIM = 384 

# --- Define AUDIO_FEATURE_DIM based on N_MFCC used in audio_feature_extraction.py ---
AUDIO_FEATURE_DIM = 128 


class TVSumDataset(Dataset):
    def __init__(self, features_dir, annotation_file, info_tsv_file, audio_features_dir, sample_rate=15):
        """
        Initializes the dataset by loading features, annotations, and video info.
        Now includes real text (title) features and real audio features (MFCCs).
        """
        self.features_dir = features_dir
        self.audio_features_dir = audio_features_dir # New attribute for audio features
        self.sample_rate = sample_rate
        self.data = [] 

        # Initialize the SentenceTransformer model (if not already loaded)
        print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded.")

        # Load annotations from .mat file
        self.video_annotations = self._load_annotations(annotation_file)
        
        # Load video info (especially titles) from .tsv file
        self.video_info = self._load_video_info(info_tsv_file)

        # Collect video names for which we have visual features
        visual_feature_files = {f.replace('_features.npy', '') for f in os.listdir(features_dir) if f.endswith('_features.npy')}
        
        # Collect video names for which we have audio features
        audio_feature_files = {f.replace('_audio_features.npy', '') for f in os.listdir(audio_features_dir) if f.endswith('_audio_features.npy')}

        print("Preparing dataset (loading visual, annotations, real text, and REAL audio info)...")
        # Iterate over videos that have BOTH visual and audio features
        for youtube_id in tqdm(sorted(list(visual_feature_files.intersection(audio_feature_files))), desc="Loading Data"):
            
            if youtube_id not in self.video_info:
                print(f"Warning: Info not found for video: {youtube_id}. Skipping.")
                continue

            visual_feature_path = os.path.join(features_dir, f"{youtube_id}_features.npy")
            audio_feature_path = os.path.join(audio_features_dir, f"{youtube_id}_audio_features.npy")

            try:
                # Load Visual Features
                visual_features = np.load(visual_feature_path)
                num_sampled_frames = visual_features.shape[0] # Number of frames after sampling visual features

                # Load Ground Truth Scores (already normalized from 0-5 and averaged across users)
                # Ensure it's flattened
                gt_score = self.video_annotations[youtube_id]['gt_score'].flatten()
                
                frame_importance = np.zeros(num_sampled_frames, dtype=np.float32)

                if len(gt_score) > 0:
                    num_original_segments = len(gt_score)
                    sampled_frames_per_original_segment = num_sampled_frames / num_original_segments

                    for i in range(num_original_segments):
                        start_idx = int(round(i * sampled_frames_per_original_segment))
                        end_idx = int(round((i + 1) * sampled_frames_per_original_segment))
                        
                        start_idx = max(0, min(start_idx, num_sampled_frames))
                        end_idx = max(0, min(end_idx, num_sampled_frames))

                        # --- FIX: Normalize gt_score from 0-5 range to 0-1 range ---
                        normalized_gt_score = gt_score[i] / 5.0 # Max score is 5 for TVSum
                        
                        if start_idx < end_idx:
                            frame_importance[start_idx:end_idx] = normalized_gt_score
                        elif start_idx < num_sampled_frames: # Handle single frame segments
                             frame_importance[start_idx] = normalized_gt_score
                
                # Clipping is still good for numerical stability, ensuring values are strictly 0-1
                frame_importance = np.clip(frame_importance, 0.0, 1.0) 

                # --- Get video title and generate REAL text feature ---
                video_title = self.video_info[youtube_id]['title']
                text_embedding = self.text_model.encode(video_title, convert_to_tensor=True)
                if text_embedding.shape[0] != TEXT_FEATURE_DIM:
                    raise ValueError(f"Text embedding dimension mismatch for video {youtube_id}. Expected {TEXT_FEATURE_DIM}, got {text_embedding.shape[0]}.")
                text_feature = text_embedding.float()
                
                # --- Load REAL audio feature and align with visual features ---
                audio_mfccs = np.load(audio_feature_path)
                
                if audio_mfccs.shape[0] != num_sampled_frames:
                    audio_time_points = np.linspace(0, 1, audio_mfccs.shape[0])
                    visual_time_points = np.linspace(0, 1, num_sampled_frames)
                    
                    aligned_audio_mfccs = np.zeros((num_sampled_frames, AUDIO_FEATURE_DIM))
                    for i in range(AUDIO_FEATURE_DIM):
                        aligned_audio_mfccs[:, i] = np.interp(visual_time_points, audio_time_points, audio_mfccs[:, i])
                    audio_feature_frame_level = torch.tensor(aligned_audio_mfccs, dtype=torch.float32)
                else:
                    audio_feature_frame_level = torch.tensor(audio_mfccs, dtype=torch.float32)

                audio_feature_video_level = audio_feature_frame_level.mean(dim=0)
                
                self.data.append({
                    'features': torch.tensor(visual_features, dtype=torch.float32),
                    'importance_scores': torch.tensor(frame_importance, dtype=torch.float32),
                    'video_name': youtube_id,
                    'text_feature': text_feature,
                    'audio_feature': audio_feature_video_level 
                })

            except Exception as e:
                print(f"Error processing video {youtube_id}: {e}. Skipping.")

        print(f"Finished dataset preparation. Total videos loaded: {len(self.data)}")


    def _load_annotations(self, annotation_file):
        annotations = {}
        with h5py.File(annotation_file, 'r') as f:
            if 'tvsum50' in f:
                tvsum_group = f['tvsum50']
                
                video_title_refs = tvsum_group['title'][()]
                user_anno_refs = tvsum_group['user_anno'][()]
                gt_score_refs = tvsum_group['gt_score'][()]
                video_id_refs = tvsum_group['video'][()]

                for i in range(len(video_title_refs)):
                    title_char_array = f[video_title_refs[i, 0]][()]
                    video_mat_title = "".join(chr(c) for c in title_char_array.flatten() if c != 0)
                    
                    youtube_id_char_array = f[video_id_refs[i, 0]][()]
                    youtube_id = "".join(chr(c) for c in youtube_id_char_array.flatten() if c != 0)
                    
                    if not youtube_id:
                        print(f"Warning: Skipping annotation from .mat for '{video_mat_title}' - extracted YouTube ID is empty. Raw ID data: {youtube_id_char_array}")
                        continue

                    gt_score = f[gt_score_refs[i, 0]][()]
                    user_anno = f[user_anno_refs[i, 0]][()]
                    
                    annotations[youtube_id] = {
                        'title_from_mat': video_mat_title,
                        'gt_score': gt_score,
                        'user_anno': user_anno,
                    }
            else:
                raise RuntimeError("Key 'tvsum50' not found in the .mat file. Check file integrity.")
        return annotations

    def _load_video_info(self, info_tsv_file):
        info = {}
        try:
            df = pd.read_csv(info_tsv_file, sep='\t')
            for _, row in df.iterrows():
                info[row['video_id']] = {
                    'category': row['category'],
                    'title': row['title'],
                    'url': row['url'],
                    'length': row['length']
                }
        except FileNotFoundError:
            raise FileNotFoundError(f"Info TSV file not found: {info_tsv_file}")
        except Exception as e:
            print(f"Error loading info TSV file {info_tsv_file}: {e}")
        return info

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (item['features'], item['importance_scores'], item['video_name'], item['text_feature'], item['audio_feature'])


def collate_fn(batch):
    """
    Collate function for DataLoader that handles real text and audio features.
    """
    features_list, scores_list, names_list, text_features_list, audio_features_list = zip(*batch)

    max_len = max([item.shape[0] for item in features_list])
    
    padded_features = torch.zeros(len(batch), max_len, features_list[0].shape[1])
    for i, features in enumerate(features_list):
        seq_len = features.shape[0]
        padded_features[i, :seq_len, :] = features
    
    padded_importance_scores = torch.zeros(len(batch), max_len)
    for i, scores in enumerate(scores_list):
        seq_len = scores.shape[0]
        padded_importance_scores[i, :seq_len] = scores
    
    stacked_text_features = torch.stack(text_features_list)
    stacked_audio_features = torch.stack(audio_features_list)

    lengths = torch.tensor([item.shape[0] for item in features_list])

    lengths, perm_idx = lengths.sort(descending=True)
    padded_features = padded_features[perm_idx]
    padded_importance_scores = padded_importance_scores[perm_idx]
    names_list = [names_list[i] for i in perm_idx]
    stacked_text_features = stacked_text_features[perm_idx]
    stacked_audio_features = stacked_audio_features[perm_idx] 

    return padded_features, padded_importance_scores, lengths, names_list, stacked_text_features, stacked_audio_features

if __name__ == '__main__':
    print("--- Testing TVSumDataset (with REAL text and REAL audio features) ---")
    
    tvsum_dataset = TVSumDataset(FEATURES_SAVE_DIR, MATLAB_ANNOTATION_FILE, INFO_TSV_FILE, AUDIO_FEATURES_SAVE_DIR)

    if len(tvsum_dataset) > 0:
        first_video_features, first_video_scores, first_video_name, first_video_text_feature, first_video_audio_feature = tvsum_dataset[0]
        print(f"\nFirst video features shape: {first_video_features.shape}")
        print(f"First video importance scores shape: {first_video_scores.shape}")
        print(f"First video name: {first_video_name}")
        print(f"First video text feature shape: {first_video_text_feature.shape}")
        print(f"First video audio feature shape: {first_video_audio_feature.shape}")

        train_dataloader = DataLoader(tvsum_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        print("\nIterating through a batch from DataLoader (now with REAL text and REAL audio features):")
        for i, (features, scores, lengths, names, text_features, audio_features) in enumerate(train_dataloader):
            print(f"Batch {i+1}:")
            print(f"  Features shape (padded): {features.shape}")
            print(f"  Scores shape (padded): {scores.shape}")
            print(f"  Original lengths: {lengths}")
            print(f"  Video Names: {names}")
            print(f"  Text Features shape: {text_features.shape}")
            print(f"  Audio Features shape: {audio_features.shape}")
            if i == 0: break
    else:
        print("Dataset is empty. Please ensure feature_extraction.py and audio_feature_extraction.py ran successfully and paths are correct.")