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

        self.criterion = nn.BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def fit(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self.train()
            self.evaluate()

    def train(self):
        self.model.train()

        train_correct = 0
        train_loss = 0
        for i, batch in enumerate(self.train_dataloader):
            print(f"{i}/{len(self.train_dataloader)}", end="\r")
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()  # clear gradients
            outputs = self.model(input_ids, attention_mask)  # forward pass
            loss = self.criterion(
                outputs,
                targets.reshape(-1, 1).float(),
            )  # calculate loss
            train_loss += loss.item()  # add loss to train_loss
            loss.backward()  # backward pass
            self.optimizer.step()  # update weights

            # calculate accuracy
            outputs = torch.round(outputs)
            train_correct += (outputs == targets.reshape(-1, 1)).float().sum()
            running_train_acc = 100 * train_correct / ((i + 1) * self.batch_size)
            print(
                f"\tLoss: {loss:4f}, Accuracy: {running_train_acc:4f}",
                end="\r",
            )

        # train_acc = 100 * train_correct / len(self.train_dataloader.dataset)

    def evaluate(self):
        with torch.no_grad():
            self.model.eval()
            val_correct = 0
            val_loss = 0
            for i, batch in enumerate(self.val_dataloader):
                print(f"{i}/{len(self.val_dataloader)}", end="\r")
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(
                    outputs,
                    targets.reshape(-1, 1).float(),
                )
                val_loss += loss.item()

                # calculate accuracy
                outputs = torch.round(outputs)
                val_correct += (outputs == targets.reshape(-1, 1)).float().sum()

        val_acc = 100 * val_correct / len(self.val_dataloader.dataset)

        print("")
        print(
            f"\tVal Loss: {val_loss/len(self.val_dataloader):4f}\t",
            f"Val Acc: {val_acc:4f}",
        )
