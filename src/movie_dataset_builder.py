import pandas as pd
from torch.utils.data import random_split
from src.movie_dataset import MovieDataset


class MovieDatasetBuilder:
    def __init__(self, filename, val_percent=0.2):
        self.filename = filename
        self.val_percent = val_percent

    def split_data(self, df):
        val_size = int(len(df) * self.val_percent)
        train_size = len(df) - val_size
        return random_split(df, [train_size, val_size])  # returns a tuple

    def build_dataset(self):
        """
        returns a pandas dataframe
        """
        # read in the data
        df = pd.read_csv(self.filename)
        print(f"\nNumber of reviews: {len(df.index)}")

        # remove duplicate rows
        df.drop_duplicates(subset=["review"], inplace=True)
        print(f"Number of unqiue reviews: {len(df.index)}\n")

        # check the distribution of the labels
        print(df.sentiment.value_counts())

        # get baseline
        # print(
        #     f"\nOur baseline accuracy is : {max(df.sentiment.value_counts(normalize=True)):.6f}"
        # )

        # convert labels to 0 and 1
        df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

        # convert df to MovieDataset
        dataset = MovieDataset(df)

        # split the data into train and validation and return
        return self.split_data(dataset)
