from src.distilbert_classifier import DistilBertClassifier
from src.training.training_session import TrainingSession
from src.training.training_args import TrainingArgs

filename = "tests/fixtures/test_reviews.csv"

default_args = TrainingArgs().parse_args()


def test_init():
    session = TrainingSession(default_args)
    assert session.filename == filename
    assert session.epochs == 3
    assert session.batch_size == 32
    assert session.learning_rate == 2e-5


def test_create_datasets():
    session = TrainingSession(default_args)
    session.create_datasets()
    assert len(session.train_dataset) == 8
    assert len(session.val_dataset) == 2


def test_create_dataloaders():
    session = TrainingSession(default_args)
    session.create_datasets()
    session.create_dataloaders()
    assert len(session.train_dataloader) == 1
    assert len(session.val_dataloader) == 1


def test_create_model():
    session = TrainingSession(default_args)
    session.create_model()
    assert isinstance(session.model, DistilBertClassifier)
