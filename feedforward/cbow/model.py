import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.output_dim = embedding_dim

        self.layer1 = nn.Linear(embedding_dim, hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim = -1)
        

    def forward(self, inputs):
        input_vector = sum(self.embeddings(inputs)).view(1, -1)

        hidden_layer = F.relu(self.layer1(input_vector))
        output_layer = self.layer2(hidden_layer)
        output = self.softmax(output_layer)
        return output

