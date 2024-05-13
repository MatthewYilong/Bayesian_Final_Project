from pyro.optim import Adam
from model import guide, model 
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


def train(train_loader, number_of_epoch):
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for epoch in range(number_of_epoch): 
        loss = 0
        for batch_idx, data in enumerate(train_loader): 
            loss += svi.step(data[0].view(-1,28*28), data[1])
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = loss / normalizer_train
        print("Epoch ", j, " Loss ", total_epoch_loss_train) 
