from re import I
from torch.utils.data import DataLoader
from src.movie_dataset_builder import MovieDatasetBuilder
from src.distilbert_classifier import DistilBertClassifier
from src.trainer import Trainer


class TrainingSession:
    def __init__(self, filename):
        self.filename = filename
        self.epochs = 3
        self.batch_size = 32
        self.learning_rate = 2e-5

    def run(self):
        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.create_trainer()
        self.trainer.fit()

    def create_datasets(self):
        builder = MovieDatasetBuilder(self.filename)
        self.train_dataset, self.val_dataset = builder.build_dataset()

    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)

    def create_model(self):
        self.model = DistilBertClassifier()

    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )


if __name__ == "__main__":
    TrainingSession("data/IMDB-Dataset.csv").run()
