import numpy as np
import torch
from torch import nn, optim
from torchtext import data, datasets
from torch import optim
from RNN import RNN

TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_md')
LABEL = data.LabelField(dtype=torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)
device = torch.device('cuda')
batchsz = 30

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size = batchsz,
    device=device
)

BCE = nn.BCEWithLogitsLoss().to(device)


rnn = RNN(len(TEXT.vocab),100,256)
pretrained_embedding = TEXT.vocab.vectors
rnn.embedding.weight.data.copy_(pretrained_embedding)
rnn.to(device)
optimizer = optim.Adam(rnn.parameters(),lr=1e-3)
avg_acc = []
acc_num = 0
acc_sum = 0

#for epoch in range(10):
#    rnn.train()
#    for i,batch in enumerate(train_iterator):
#        pred = rnn(batch.text).squeeze(1)
#        loss = BCE(pred,batch.label)
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#        print(i,  loss.item())
#
rnn.load_state_dict(torch.load("RNN.pt"))
rnn.eval()
with torch.no_grad():
    for i,batch in enumerate(test_iterator):
        pred = rnn(batch.text).squeeze(1)
        pred = torch.round(torch.sigmoid(pred))
        acc_sum += torch.eq(pred,batch.label).float().sum().item()
        acc_num += batch.label.size(0)
    acc = acc_sum/acc_num
    avg_acc.append(acc)
    #torch.save(rnn.state_dict(),'RNN.pt')
    print(acc)

# Final Correct Rate = 0.87796
