import torch
import torch.nn as nn

from tqdm import tqdm

from dataset import process_text
from model import Skipgram

CONTEXT_SIZE = 2
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 128
LEARNING_RATE = 0.001

data, word_to_ix, ix_to_word, vocab_size = process_text(CONTEXT_SIZE)

model = Skipgram(EMBEDDING_SIZE, HIDDEN_SIZE, vocab_size, CONTEXT_SIZE)

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


for epoch in range(100):
    with tqdm(data, unit="batch") as tepoch:
        total_loss = 0

        for _context, _input in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            target_vector = torch.tensor(word_to_ix[_input], dtype=torch.long)
            context_vectors = [word_to_ix[w] for w in _context]
            print(target_vector)
            print(context_vectors)
            
            y_pred = model(target_vector)
            print("###")
            print(y_pred)
            print(y_pred.size())
            print(target_vector)
            loss = nn.CrossEntropyLoss(y_pred, context_vectors)
            print(loss)
            total_loss += loss
            tepoch.set_postfix(loss=loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

#TESTING
input_ = 'model'
context = model(input_)

#Print result
print(f'Context: {context}\n')
print(f'Prediction: {[ix_to_word[torch.argmax(a[0]).item()] for a in context]}')