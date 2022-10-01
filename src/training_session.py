from torch.utils.data import DataLoader
from src.movie_dataset_builder import MovieDatasetBuilder
from src.distilbert_classifier import DistilBertClassifier
from src.trainer import Trainer


class TrainingSession:
    def __init__(self, filename):
        self.filename = filename
        self.epochs = 5
        self.batch_size = 64
        self.learning_rate = 2e-5

    def run(self):
        builder = MovieDatasetBuilder(self.filename)
        train_dataset, val_dataset = builder.build_dataset()

        model = DistilBertClassifier()

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )
        trainer.fit()


if __name__ == "__main__":
    TrainingSession("data/IMDB-Dataset.csv").run()
