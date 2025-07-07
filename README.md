# Video Summarization using Deep Learning (Multimodal LSTM)

## Project Objective

The primary objective of this project is to develop a prototype deep learning model capable of generating concise summaries of videos by automatically identifying and selecting key scenes. The approach leverages multimodal information (visual, audio, and text) to enhance the summarization process.

## Requirements Fulfilled

* **Dataset Usage:** Utilized the **TVSum dataset** for training and evaluation.
* **Model Implementation:** Implemented a video summarization model based on a **Long Short-Term Memory (LSTM)** network for robust temporal modeling.
* **Multimodal Feature Extraction:** Incorporated three distinct modalities:
    * **Visual Features:** Extracted frame-wise features using a pre-trained **ResNet-50** CNN.
    * **Audio Features:** Extracted **Mel-Frequency Cepstral Coefficients (MFCCs)** to represent audio content.
    * **Text Features:** Utilized **SentenceTransformer ('all-MiniLM-L6-v2')** to generate embeddings from video titles.
* **Evaluation:** The summary output is quantitatively evaluated against the dataset's provided ground-truth importance scores using standard metrics: **Precision, Recall, and F1-score**.

## Tasks Completed

1.  **Dataset Preprocessing & Feature Extraction:**
    * Developed scripts to extract frame-wise features from videos:
        * **Visual:** ResNet-50 features from sampled video frames.
        * **Audio:** MFCCs from audio segments, aligned with visual frames.
        * **Text:** Embeddings of video titles using SentenceTransformer.
    * Implemented a custom PyTorch `TVSumDataset` to load and prepare these multimodal features along with ground-truth importance scores.
    * Handled video-level ground truth scores and interpolated them to frame-level.
    * Corrected ground truth normalization to map scores from 0-5 to 0-1, providing accurate importance targets.

2.  **Summarization Model Implementation:**
    * Designed and implemented a `VideoSummarizerLSTM` model in PyTorch.
    * The model takes concatenated visual, text (video-level), and audio (video-level) features as input to an LSTM layer, predicting an importance score for each frame.

3.  **Multimodal Incorporation:**
    * Successfully incorporated **visual, text, and audio** modalities. Features from all three are combined (concatenated) and fed into the LSTM, allowing the model to learn from diverse information streams.

4.  **Summary Generation & Comparison:**
    * The `evaluate.py` script generates frame-level importance scores.
    * A summary is created by selecting the **top K% of frames** based on their predicted importance scores (where K is set to 15%).
    * The generated summary (binary sequence) is then compared against the ground truth (also binarized by taking its top K% most important frames) to compute Precision, Recall, and F1-score.

5.  **Visual Output:**
    * The `evaluate.py` script is configured to identify and print the indices of the selected keyframes for each video.
    * It also **simulates the saving of these selected keyframes** into a dedicated `summary_frames` directory for each video, assuming the original video files are accessible. *To fully realize the visual output, you would need to implement the actual frame extraction and saving logic using libraries like OpenCV (`cv2`).*

## Bonus Features Explored

* **Multi-modal Fusion (Visual + Audio + Text):** Achieved by concatenating the extracted features from all three modalities before feeding them into the LSTM. This allows the model to learn joint representations.
* **Attention Mechanisms:** While not explicitly implemented as a separate attention layer in the final LSTM model, this remains a strong candidate for future work to further improve scene importance estimation.

## Deliverables

* **Python Scripts:**
    * `feature_extraction.py`: Extracts ResNet visual features from videos.
    * `audio_feature_extraction.py`: Extracts MFCC audio features from videos.
    * `dataset.py`: PyTorch `Dataset` and `collate_fn` for multimodal data loading.
    * `model.py`: Defines the `VideoSummarizerLSTM` architecture.
    * `train.py`: Script for training the model.
    * `evaluate.py`: Script for evaluating the trained model and generating summaries.
* **Model Weights:** The trained model weights (`summarizer_model_best_val_loss.pth`) are saved in the `checkpoints/` directory after successful training.
* **README File:** This document.
* **Evaluation Metrics Report:** Provided in the console output by `evaluate.py` (e.g., Precision, Recall, F1-score). The current F-score is approximately **0.3913**.

## Setup Instructions

Follow these steps to set up the project, extract features, train, and evaluate the model.

### 1. Project Structure

Ensure your project directory is organized as follows:
