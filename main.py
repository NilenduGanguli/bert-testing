import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast

# Specify GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BERT tokenizer and pre-trained model (bert-base-uncased)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased')

# Example text for processing
text = ["This is an example sentence.", "BERT is a powerful model."]

# Tokenize and encode the sequences
# truncation=True ensures sequences are limited to max length (512 for BERT)
# padding='max_length' pads shorter sequences
tokens = tokenizer._encode_plus(
    text,
    max_length=25,
    padding='max_length',
    truncation=True,
    return_tensors='pt' # Return PyTorch tensors
)

# Move tensors to the specified device
train_seq = tokens['input_ids'].to(device)
train_mask = tokens['attention_mask'].to(device)

# Example of a simple classification model architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512) # 768 is the hidden size for BERT-base
        self.fc2 = nn.Linear(512, 2)   # 2 for binary classification
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        # Pass inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Instantiate the model and move to device (for fine-tuning, you would train this model)
model = BERT_Arch(bert)
model = model.to(device)

print("\nToken IDs (Input IDs):\n", train_seq)
print("\nAttention Masks:\n", train_mask)
print("\nModel Architecture:\n", model)
