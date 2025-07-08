import torch
import torch.nn as nn

# Import the updated TEXT_FEATURE_DIM and AUDIO_FEATURE_DIM from dataset.py
from dataset import TEXT_FEATURE_DIM, AUDIO_FEATURE_DIM 

class VideoSummarizerLSTM(nn.Module):
    def __init__(self, visual_feature_dim, text_feature_dim, audio_feature_dim, hidden_dim, num_layers=1, dropout=0.0, bidirectional=False):
        """
        Initializes the LSTM-based video summarization model, now with multimodal input
        including visual, text, and audio features.

        Args:
            visual_feature_dim (int): The dimensionality of visual input features (e.g., 2048 for ResNet-50).
            text_feature_dim (int): The dimensionality of text input features (e.g., 384).
            audio_feature_dim (int): The dimensionality of audio input features (e.g., 128 for VGGish).
            hidden_dim (int): The size of the LSTM hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability for regularization.
            bidirectional (bool): If True, becomes a bidirectional LSTM.
        """
        super(VideoSummarizerLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.visual_feature_dim = visual_feature_dim
        self.text_feature_dim = text_feature_dim
        self.audio_feature_dim = audio_feature_dim
        
        # The total input dimension to the LSTM will be the sum of visual, text, and audio feature dimensions
        # This will now correctly use 384 for TEXT_FEATURE_DIM
        total_input_dim = visual_feature_dim + text_feature_dim + audio_feature_dim

        # LSTM layer to process sequential combined features
        self.lstm = nn.LSTM(total_input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)

        # The output of a bidirectional LSTM will have hidden_dim * 2 features
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Fully connected layer to map LSTM output to a single importance score per frame
        self.fc = nn.Linear(fc_input_dim, 1)

        self.sigmoid = nn.Sigmoid() 

    def forward(self, visual_features, lengths, text_features, audio_features):
        """
        Forward pass through the model.

        Args:
            visual_features (torch.Tensor): Padded visual features of shape (batch_size, max_seq_len, visual_feature_dim).
            lengths (torch.Tensor): A 1D tensor containing the actual lengths of each sequence in the batch.
                                    These must be sorted in decreasing order for pack_padded_sequence.
            text_features (torch.Tensor): Text features for each video in the batch, shape (batch_size, text_feature_dim).
                                         These are video-level features.
            audio_features (torch.Tensor): Audio features for each video in the batch, shape (batch_size, audio_feature_dim).
                                          These are video-level features.
        Returns:
            torch.Tensor: Predicted importance scores for each frame, shape (batch_size, max_seq_len).
        """
        batch_size, max_seq_len, _ = visual_features.shape

        # Expand text_features to match the sequence length of visual_features
        expanded_text_features = text_features.unsqueeze(1).expand(-1, max_seq_len, -1)

        # Expand audio_features to match the sequence length of visual_features
        expanded_audio_features = audio_features.unsqueeze(1).expand(-1, max_seq_len, -1)

        # Concatenate visual, expanded text, and expanded audio features along the last dimension
        combined_features = torch.cat((visual_features, expanded_text_features, expanded_audio_features), dim=2)

        # Pack the padded sequences.
        packed_features = nn.utils.rnn.pack_padded_sequence(combined_features, lengths.cpu(), batch_first=True)

        # Pass packed features through the LSTM
        lstm_out, (h_n, c_n) = self.lstm(packed_features)

        # Unpack the output sequence.
        unpacked_lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Apply the fully connected layer to each time step's output
        fc_out = self.fc(unpacked_lstm_out)

        # Apply sigmoid to get scores between 0 and 1, and remove the last dimension
        predicted_scores = self.sigmoid(fc_out).squeeze(-1)

        return predicted_scores

# Example usage (for testing the model structure directly)
if __name__ == '__main__':
    # Define dummy parameters for testing
    dummy_visual_feature_dim = 2048
    dummy_text_feature_dim = TEXT_FEATURE_DIM # Will now be 384
    dummy_audio_feature_dim = AUDIO_FEATURE_DIM 
    dummy_hidden_dim = 512
    dummy_num_layers = 2
    dummy_dropout = 0.5
    dummy_bidirectional = True

    # Create a dummy batch of features with variable lengths
    # BATCH_SIZE = 3, MAX_SEQ_LEN = 10
    
    # Dummy visual features (padded)
    dummy_visual_features = torch.randn(3, 10, dummy_visual_feature_dim)
    
    # Dummy text features (one per video in batch), now matching TEXT_FEATURE_DIM from dataset.py
    dummy_text_features = torch.randn(3, dummy_text_feature_dim)

    # Dummy audio features (one per video in batch)
    dummy_audio_features = torch.randn(3, dummy_audio_feature_dim)

    dummy_lengths_unsorted = torch.tensor([8, 5, 10], dtype=torch.long)

    # --- CRITICAL: Sort lengths and reorder features accordingly ---
    dummy_lengths, perm_idx = dummy_lengths_unsorted.sort(descending=True)
    dummy_visual_features = dummy_visual_features[perm_idx]
    dummy_text_features = dummy_text_features[perm_idx]
    dummy_audio_features = dummy_audio_features[perm_idx]
    # --- END FIX ---

    # Instantiate the model - now passing audio_feature_dim
    model = VideoSummarizerLSTM(dummy_visual_feature_dim, dummy_text_feature_dim, dummy_audio_feature_dim, 
                                dummy_hidden_dim, num_layers=dummy_num_layers, 
                                dropout=dummy_dropout, bidirectional=dummy_bidirectional)

    print("Model structure:")
    print(model)

    # Move model and dummy data to a device if available (e.g., GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_visual_features = dummy_visual_features.to(device)
    dummy_text_features = dummy_text_features.to(device)
    dummy_audio_features = dummy_audio_features.to(device)

    print(f"\nTesting forward pass on {device}...")
    predictions = model(dummy_visual_features, dummy_lengths, dummy_text_features, dummy_audio_features)

    print(f"Input visual features shape: {dummy_visual_features.shape}")
    print(f"Input text features shape: {dummy_text_features.shape}")
    print(f"Input audio features shape: {dummy_audio_features.shape}")
    print(f"Input lengths: {dummy_lengths}")
    print(f"Predicted scores shape: {predictions.shape}")
    print(f"Predicted scores (first video, first 5 frames):\n{predictions[0, :5].detach().cpu().numpy()}")
    print("Forward pass successful with multimodal (visual + text + audio) input!")