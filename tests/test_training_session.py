from transformers import DistilBertModel
from src.training_session import TrainingSession


def test_init():
    filename = "tests/fixtures/test_reviews.csv"
    session = TrainingSession(filename)
    assert session.filename == filename
    assert session.epochs == 3
    assert session.batch_size == 32
    assert session.learning_rate == 2e-5


def test_create_datasets():
    filename = "tests/fixtures/test_reviews.csv"
    session = TrainingSession(filename)
    session.create_datasets()
    assert len(session.train_dataset) == 8
    assert len(session.val_dataset) == 2


def test_create_dataloaders():
    filename = "tests/fixtures/test_reviews.csv"
    session = TrainingSession(filename)
    session.create_datasets()
    session.create_dataloaders()
    assert len(session.train_dataloader) == 1
    assert len(session.val_dataloader) == 1


def test_create_model():
    filename = "tests/fixtures/test_reviews.csv"
    session = TrainingSession(filename)
    session.create_model()
    assert type(session.model.distilbert_model) == DistilBertModel
