import torch
import torch.nn as nn
import torch.nn.functional as F


class Skipgram(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, context_size):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_size = context_size
        self.hidden_dim = hidden_dim
        self.output_dim = vocab_size * context_size

        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, self.output_dim)
        self.softmax = nn.LogSoftmax(dim = -1)
        

    def forward(self, input):
        embedding = self.embeddings(input).view(1, -1)
        hidden_layer = F.relu(self.layer1(embedding))
        output_layer = self.layer2(hidden_layer)
        output = self.softmax(output_layer).view(self.context_size, -1)
        return output

