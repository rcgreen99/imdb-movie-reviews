# training loop
import time
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
        print("Training...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self.train()
            self.evaluate()

    def estimated_time_remaining(self, curr_batch, total_batches, avg_time_per_batch):
        batches_left = total_batches - curr_batch
        time_left = batches_left * avg_time_per_batch
        return time_left

    def train(self):
        self.model.train()

        train_correct = 0
        running_train_loss = 0
        average_batch_time = 0
        for i, batch in enumerate(self.train_dataloader):
            training_time = time.time()
            print(f"{i + 1}/{len(self.train_dataloader)}", end="\r")

            input_ids, attention_mask, targets = self.unpack_batch(batch)

            self.optimizer.zero_grad()  # clear gradients
            outputs = self.model(input_ids, attention_mask)  # forward pass
            loss, train_loss = self.calculate_loss(outputs, targets)
            loss.backward()  # backward pass
            self.optimizer.step()  # update weights

            # calculate running accuracy
            outputs = torch.round(outputs)
            train_correct += (outputs == targets.reshape(-1, 1)).float().sum()
            running_train_acc = 100 * train_correct / ((i + 1) * self.batch_size)

            # calculate running loss
            running_train_loss += loss.item() * ((i + 1) * self.batch_size)

            # calculate time remaining
            average_batch_time = (
                (average_batch_time * (i)) + (time.time() - training_time)
            ) / (i + 1)
            remaining_time = self.estimated_time_remaining(
                i, len(self.train_dataloader), average_batch_time
            )

            print(
                f"\t - {time.time() - training_time:.3f}s - Loss: {loss:.6f}\tAccuracy: {running_train_acc:.6f} \tEst time: {remaining_time:.2f}s",
                end="\r",
            )

        print("")

        # train_acc = 100 * train_correct / len(self.train_dataloader.dataset)

    def evaluate(self):
        with torch.no_grad():
            self.model.eval()
            val_correct = 0
            val_loss = 0
            for i, batch in enumerate(self.val_dataloader):
                input_ids, attention_mask, targets = self.unpack_batch(batch)

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
        print(
            f"Val Loss: {val_loss/len(self.val_dataloader):.6f}\t",
            f"Val Acc: {val_acc:.6f}",
        )

    def save_model_state(self, path):
        torch.save(self.model.state_dict(), path)

    def unpack_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        targets = batch["target"].to(self.device)
        return input_ids, attention_mask, targets

    def calculate_loss(self, outputs, targets):
        # calculate loss
        loss = self.criterion(
            outputs,
            targets.reshape(-1, 1).float(),
        )
        train_loss += loss.item()  # add loss to train_loss
        return loss, train_loss
