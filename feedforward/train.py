import torch
import torch.nn as nn

from tqdm import tqdm

from dataset import process_text
from model import CBOW

CONTEXT_SIZE = 2
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001

data, word_to_ix, ix_to_word, vocab_size = process_text(CONTEXT_SIZE)

model = CBOW(EMBEDDING_SIZE, HIDDEN_SIZE, vocab_size)

loss = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


for epoch in tqdm(range(100)):
    total_loss = 0

    for context, target in data:
        context_vector = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        target_index = word_to_ix[target]
        
        y_pred = model(context_vector)
        total_loss += loss(y_pred, torch.tensor([target_index], dtype=torch.long))

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

#TESTING
context = ['highest','overall', 'and','consistently']
context_vector = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
a = model(context_vector)

#Print result
print(f'Context: {context}\n')
print(f'Prediction: {ix_to_word[torch.argmax(a[0]).item()]}')