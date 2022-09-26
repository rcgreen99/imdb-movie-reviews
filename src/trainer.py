# training loop
import re
import torch
import torch.nn as nn
from torch.optim import Adam


class Trainer:
    def __init__(
        self, model, train_dataloader, val_dataloader, learning_rate, epochs, batch_size
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        print("Training started...")

        # use_cuda = torch.cuda.is_available()
        # device = torch.device("cuda" if use_cuda else "cpu")
        device = torch.device("cpu")

        criterion = nn.BCELoss()
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        # if use_cuda:
        #     model = model.cuda()
        #     criterion = criterion.cuda()

        self.epochs = 1
        for epoch in range(self.epochs):
            self.model.train()
            print(f"Starting epoch {epoch + 1} of {self.epochs}")
            train_acc = 0
            train_loss = 0
            for i, batch in enumerate(self.train_dataloader):
                print(f"Starting batch {i + 1} of {len(self.train_dataloader)}")
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)

                optimizer.zero_grad()  # clear gradients
                outputs = self.model(input_ids, attention_mask)  # forward pass
                loss = criterion(
                    outputs,
                    targets.reshape(-1, 1).float(),  # not sure what this is doing***
                )  # calculate loss
                train_loss += loss.item()  # add loss to train_loss
                loss.backward()  # backward pass
                optimizer.step()  # update weights

            val_acc = 0
            val_loss = 0
            with torch.no_grad():
                self.model.eval()
                for batch in self.val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    targets = batch["targets"].to(device)

                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            print(f"Epoch: {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_loss/len(self.train_dataloader)}")
            print(f"Val Loss: {val_loss/len(self.val_dataloader)}")
