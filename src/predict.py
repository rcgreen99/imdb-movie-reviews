import sys
import torch
from transformers import DistilBertTokenizerFast
from src.distilbert_classifier import DistilBertClassifier


def predict(model, input):
    model.eval()
    input_ids = input["input_ids"].unsqueeze(0).to("cuda")
    atten_mask = input["attention_mask"].to("cuda")
    output = model(input_ids=input_ids, attention_mask=atten_mask)

    print(output)
    prediction = torch.round(output)
    return prediction.item()


if __name__ == "__main__":
    review = sys.argv[1]
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    encoded_review = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    input = {
        "input_ids": encoded_review["input_ids"].flatten(),
        "attention_mask": encoded_review["attention_mask"].flatten(),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertClassifier()
    model.load_state_dict(torch.load("models/model_e1.pth"))
    model = model.to(device)

    prediction = predict(model, input)
    print(f"Prediction: {prediction}")
