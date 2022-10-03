# build dataset class for our movie dataset
import torch
from transformers import DistilBertTokenizerFast
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, dataframe, max_len=512):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        self.data = dataframe
        self.reviews = dataframe.review
        self.targets = dataframe.sentiment
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        review = str(self.reviews[index])
        target = self.targets[index]

        encoded_review = self.encode_review(review)

        return {
            # "review_text": review,  # for debugging
            "input_ids": encoded_review["input_ids"].flatten(),
            "attention_mask": encoded_review["attention_mask"].flatten(),
            "target": torch.tensor(target),
        }

    def encode_review(self, review_text):
        encoded_review = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return encoded_review
