import sys
import torch
from transformers import DistilBertTokenizerFast
from src.distilbert_classifier import DistilBertClassifier


class Predictor:
    def __init__(self, model_path):
        self.model = DistilBertClassifier()
        self.model.load_state_dict(torch.load(model_path))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

    def predict(self, review):
        self.model.eval()

        input = self.preprocess(review)

        input_ids = input["input_ids"].unsqueeze(0).to(self.device)
        atten_mask = input["attention_mask"].unsqueeze(0).to(self.device)

        output = self.model(input_ids=input_ids, attention_mask=atten_mask)
        output_val = str(output.item())[:6]
        prediction = int(torch.round(output))

        return output_val, prediction

    def preprocess(self, review):
        encoded_review = self.tokenizer.encode_plus(
            review,
            max_length=512,
            add_special_tokens=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        input = {
            "input_ids": encoded_review["input_ids"].flatten(),
            "attention_mask": encoded_review["attention_mask"].flatten(),
        }
        return input


if __name__ == "__main__":
    model_path = sys.argv[1]
    review = sys.argv[2]

    predictor = Predictor(model_path)
    output, prediction = predictor.predict(review)

    print(f"Prediction: {prediction}")
