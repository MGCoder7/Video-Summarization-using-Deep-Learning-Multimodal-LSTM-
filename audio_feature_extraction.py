import os
import librosa
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import warnings

# --- Set explicit FFmpeg paths for pydub ---
# IMPORTANT: Ensure these paths are correct for your system
FFMPEG_EXE_PATH = r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"
FFPROBE_EXE_PATH = r"C:\Program Files\ffmpeg\bin\ffprobe.exe"

# Check if the executables exist before setting, to provide a helpful warning
if not os.path.exists(FFMPEG_EXE_PATH):
    print(f"Warning: ffmpeg.exe not found at {FFMPEG_EXE_PATH}. Please check the path.")
    print("pydub might still try to find it in system PATH or fail.")
if not os.path.exists(FFPROBE_EXE_PATH):
    print(f"Warning: ffprobe.exe not found at {FFPROBE_EXE_PATH}. Please check the path.")
    print("pydub might still try to find it in system PATH or fail.")

# Set environment variables for pydub to pick up
os.environ["FFMPEG_PATH"] = FFMPEG_EXE_PATH
os.environ["FFPROBE_PATH"] = FFPROBE_EXE_PATH

# Suppress librosa warnings if they are too noisy
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

# --- Configuration ---
VIDEO_DIR = r'C:\Users\Omen\OneDrive\Documents\AI\AgenticAIWorkspace\video_summarization_project\ydata-tvsum50-v1_1\ydata-tvsum50-video' # Directory containing original video files (mp4)
AUDIO_FEATURES_SAVE_DIR = 'notebooks/data/extracted_audio_features'
SAMPLE_RATE_AUDIO_EXTRACT = 16000 # Sample rate for audio extraction and MFCC calculation
N_MFCC = 128 # Number of MFCCs to extract
FPS_SAMPLING = 15 # Frame rate at which visual features were sampled (from feature_extraction.py)

# Create directory for saving audio features
os.makedirs(AUDIO_FEATURES_SAVE_DIR, exist_ok=True)

print("Starting audio feature extraction...")

# List all video files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]

for video_filename in tqdm(video_files, desc="Extracting Audio Features"):
    video_path = os.path.join(VIDEO_DIR, video_filename)
    video_id = os.path.splitext(video_filename)[0] # e.g., 'video123' from 'video123.mp4'
    
    audio_features_path = os.path.join(AUDIO_FEATURES_SAVE_DIR, f"{video_id}_audio_features.npy")

    if os.path.exists(audio_features_path):
        # print(f"Audio features for {video_id} already exist. Skipping.")
        continue

    try:
        # 1. Load video's audio using pydub
        audio = AudioSegment.from_file(video_path)
        
        # Convert to mono and target sample rate if necessary
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != SAMPLE_RATE_AUDIO_EXTRACT:
            audio = audio.set_frame_rate(SAMPLE_RATE_AUDIO_EXTRACT)

        # Convert pydub AudioSegment to a numpy array for librosa
        audio_data = np.array(audio.get_array_of_samples()).astype(np.float32)
        # For pydub, samples are integers, need to normalize them to float for librosa
        audio_data = audio_data / (1 << (audio.sample_width * 8 - 1)) # Normalize to range [-1.0, 1.0]

        # 2. Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE_AUDIO_EXTRACT, n_mfcc=N_MFCC)
        
        # 3. Time-align MFCCs with video frames
        duration_sec = len(audio) / 1000.0 # pydub length is in milliseconds
        num_visual_frames = int(np.ceil(duration_sec * FPS_SAMPLING))
        
        mfcc_time_points = np.linspace(0, duration_sec, mfccs.shape[1])
        frame_time_points = np.linspace(0, duration_sec, num_visual_frames)
        
        aligned_mfccs = np.zeros((num_visual_frames, N_MFCC))
        
        for i in range(N_MFCC):
            aligned_mfccs[:, i] = np.interp(frame_time_points, mfcc_time_points, mfccs[i, :])

        # 4. Save the aligned MFCCs
        np.save(audio_features_path, aligned_mfccs)
        
    except Exception as e:
        print(f"Error processing {video_id}: {e}. Skipping audio feature extraction for this video.")

print("Audio feature extraction complete.")
print(f"Features saved to: {AUDIO_FEATURES_SAVE_DIR}")