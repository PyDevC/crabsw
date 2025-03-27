import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the OCR model class
class OCRModel(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Input: 1 channel (grayscale)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),       # H: 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),       # H: 16 -> 8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),       # H: 8 -> 4, W reduced less
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),       # H: 4 -> 2
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),# H: 2 -> 1
            nn.ReLU()
        )
        
        # RNN for sequence modeling
        self.rnn = nn.LSTM(input_size=512, hidden_size=hidden_size, 
                          num_layers=1, bidirectional=True)
        
        # Fully connected layer to output character probabilities
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 because bidirectional
        
    def forward(self, x):
        # Input: (batch_size, channels=1, height=32, width)
        # CNN forward pass
        features = self.cnn(x)  # (batch_size, 512, 1, width')
        
        # Squeeze the height dimension
        features = features.squeeze(2)  # (batch_size, 512, width')
        
        # Permute for RNN: (width', batch_size, 512)
        features = features.permute(2, 0, 1)
        
        # RNN forward pass
        rnn_output, _ = self.rnn(features)  # (width', batch_size, hidden_size * 2)
        
        # Fully connected layer
        output = self.fc(rnn_output)  # (width', batch_size, num_classes)
        
        # Apply log softmax for CTC loss
        log_probs = F.log_softmax(output, dim=2)  # (width', batch_size, num_classes)
        
        return log_probs

# Example usage
def train():
    # Character set (example: lowercase letters + blank token)
    chars = "abcdefghijklmnopqrstuvwxyz" + " "  # blank token is index 0
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 0 reserved for blank
    num_classes = len(chars) + 1  # +1 for blank token
    
    # Initialize model
    model = OCRModel(num_classes=num_classes)
    
    # Example input (batch of grayscale images)
    batch_size = 2
    height, width = 32, 100  # Fixed height, variable width
    dummy_input = torch.randn(batch_size, 1, height, width)
    
    # Forward pass
    log_probs = model(dummy_input)  # (width', batch_size, num_classes)
    print(f"Output shape: {log_probs.shape}")
    
    # CTC Loss (during training)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Dummy targets and lengths
    targets = torch.tensor([1, 2, 3, 4, 5, 1, 2], dtype=torch.long)  # "abcde" and "ab"
    input_lengths = torch.tensor([log_probs.size(0)] * batch_size, dtype=torch.long)
    target_lengths = torch.tensor([5, 2], dtype=torch.long)  # Lengths of "abcde" and "ab"
    
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    print(f"CTC Loss: {loss.item()}")