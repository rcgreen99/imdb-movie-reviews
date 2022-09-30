# training loop
import re
import torch
import torch.nn as nn
from torch.optim import Adam


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        learning_rate,
        epochs,
        batch_size,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        """
        trains the model and evaluates it on the validation set
        """
        print("Training started...")

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        criterion = nn.BCELoss()
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        if use_cuda:
            self.model = self.model.cuda()
            criterion = criterion.cuda()

        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch + 1} of {self.epochs}")
            self.model.train()

            train_correct = 0
            train_loss = 0
            for i, batch in enumerate(self.train_dataloader):
                if i % 100 == 0:
                    print(f"Training batch {i} of {len(self.train_dataloader)}")
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target"].to(device)

                optimizer.zero_grad()  # clear gradients
                outputs = self.model(input_ids, attention_mask)  # forward pass
                loss = criterion(
                    outputs,
                    targets.reshape(-1, 1).float(),  # not sure what this is doing***
                )  # calculate loss
                train_loss += loss.item()  # add loss to train_loss
                loss.backward()  # backward pass
                optimizer.step()  # update weights

                # calculate accuracy
                outputs = torch.round(outputs)
                train_correct += (outputs == targets.reshape(-1, 1)).float().sum()

            train_acc = 100 * train_correct / len(self.train_dataloader.dataset)

            val_correct = 0
            val_loss = 0
            with torch.no_grad():
                self.model.eval()
                for i, batch in enumerate(self.val_dataloader):
                    if i % 100 == 0:
                        print(f"Evaluating Batch {i} of {len(self.val_dataloader)}")
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    targets = batch["target"].to(device)

                    loss = criterion(
                        outputs,
                        targets.reshape(
                            -1, 1
                        ).float(),  # not sure what this is doing***
                    )  # calculate loss
                    val_loss += loss.item()

                    # calculate accuracy
                    outputs = torch.round(outputs)
                    val_correct += (outputs == targets.reshape(-1, 1)).float().sum()

            val_acc = 100 * val_correct / len(self.val_dataloader.dataset)

            print(f"Epoch: {epoch+1}/{self.epochs}")
            print(f"Train Loss: {train_loss/len(self.train_dataloader):4f}")
            print(f"Train Acc: {train_acc:4f}")
            print(f"Val Loss: {val_loss/len(self.val_dataloader):4f}")
            print(f"Val Acc: {val_acc:4f}")
