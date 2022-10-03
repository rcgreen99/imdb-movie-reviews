from torch.utils.data import DataLoader
from src.movie_dataset_builder import MovieDatasetBuilder
from src.distilbert_classifier import DistilBertClassifier
from src.training.training_args import TrainingArgs
from src.training.trainer import Trainer


class TrainingSession:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.create_datasets()
        self.create_dataloaders()
        self.create_model()
        self.create_trainer()
        self.trainer.fit()

    def create_datasets(self):
        builder = MovieDatasetBuilder(self.args.filename)
        self.train_dataset, self.val_dataset = builder.build_dataset()

    def create_dataloaders(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size
        )

    def create_model(self):
        self.model = DistilBertClassifier()

    def create_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            learning_rate=self.args.learning_rate,
            epochs=self.args.epochs,
        )


if __name__ == "__main__":
    args = TrainingArgs().parse_args()
    session = TrainingSession(args)
    session.run()
