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

Okay, I see the updated project structure from your screenshot. This helps clarify the exact locations of your dataset files and scripts.

Here's the revised README.md content, with the "Project Structure" section updated to match your screenshot. I've also adjusted the "Download Dataset" section slightly to reflect where the video files are likely located within ydata-tvsum50-v1_1.

README.md (Updated Project Structure)
Markdown

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
    * `debug_mat_structure.py` (Utility script for exploring .mat file structure)
* **Model Weights:** The trained model weights (`summarizer_model_best_val_loss.pth`) are saved in the `checkpoints/` directory after successful training.
* **README File:** This document.
* **Evaluation Metrics Report:** Provided in the console output by `evaluate.py` (e.g., Precision, Recall, F1-score). The current F-score is approximately **0.3913**.

## Setup Instructions

Follow these steps to set up the project, extract features, train, and evaluate the model.

### 1. Project Structure

Ensure your project directory is organized as follows:

VIDEO_SUMMARIZATION_PROJECT/
├── checkpoints/             # Saved model weights will be stored here
├── notebooks/
│   └── 00_data_exploration.ipynb # Jupyter notebook for initial data exploration
├── data/
│   ├── extracted_features/  # Output of feature_extraction.py (ResNet features)
│   └── extracted_audio_features/ # Output of audio_feature_extraction.py (MFCC features)
├── summary_frames/          # Generated summary frames will be saved here
├── ydata-tvsum50-v1_1/
│   ├── ydata-tvsum50-matlab/ # TVSum ground truth annotations (.mat file inside)
│   ├── ydata-tvsum50-thumbnail/
│   ├── ydata-tvsum50-video/ # Place your TVSum50 .mp4 video files here
│   └── README               # TVSum dataset's README
├── ydata-tvsum50-anno.tsv   # Additional TVSum annotation file (not directly used in current pipeline)
├── ydata-tvsum50-info.tsv   # TVSum video info (titles, etc.)
├── audio_feature_extraction.py
├── dataset.py
├── debug_mat_structure.py   # Utility script
├── evaluate.py
├── feature_extraction.py
├── model.py
├── train.py
├── requirements.txt         # (Assumed to be at root level)
└── README.md                # This file

### 2. Download Dataset

1.  **TVSum Dataset:** Download the TVSum dataset. You'll need:
    * The `ydata-tvsum50-v1_1.zip` file (contains `ydata-tvsum50-matlab`, `ydata-tvsum50-thumbnail`, `ydata-tvsum50-video`, and the dataset's `README`).
    * The `ydata-tvsum50-info.tsv` file (and optionally `ydata-tvsum50-anno.tsv`).
2.  **Extract:**
    * Extract the contents of `ydata-tvsum50-v1_1.zip` into a directory named `ydata-tvsum50-v1_1/` in your project root. Ensure the `.mp4` video files are within `ydata-tvsum50-v1_1/ydata-tvsum50-video/`.
    * Place `ydata-tvsum50-info.tsv` (and `ydata-tvsum50-anno.tsv` if included) directly in your project root.
    * **Adjust paths:** Make sure the paths in `dataset.py`, `feature_extraction.py`, and `audio_feature_extraction.py` (e.g., `TVPARM_BASE_DIR`, `MATLAB_ANNOTATION_FILE`, `INFO_TSV_FILE`, `VIDEO_DIR`) correctly point to your extracted dataset files according to the new structure.

### 3. Set Up Environment

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd VIDEO_SUMMARIZATION_PROJECT # Or whatever your project root directory is named
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 4. Run the Pipeline

Execute the following scripts in order:

#### a. Extract Visual Features

This will process all `.mp4` videos found in `ydata-tvsum50-v1_1/ydata-tvsum50-video/` and save ResNet-50 features to `data/extracted_features/`.

```bash
python feature_extraction.py
```

#### b. Extract Audio Features

This will process all .mp4 videos found in ydata-tvsum50-v1_1/ydata-tvsum50-video/ and save MFCC features to data/extracted_audio_features/.
```bash
python audio_feature_extraction.py
```

#### c.  Train the Model

This will train the multimodal LSTM model. The best model (based on validation loss) will be saved in checkpoints/.
```bash
python train.py
```

#### d. Evaluate the Model

This will load the best-trained model, evaluate its performance on the dataset, print evaluation metrics (Precision, Recall, F1-score), and simulate saving summary frames.
```bash
python evaluate.py
```


