from transformers import DistilBertModel, DistilBertForSequenceClassification
from torch import nn

# create custom model
class DistilBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.distilbert_model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.dense1 = nn.Linear(768, 768)
        self.relu1 = nn.ReLU()
        # self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # _, pooled_output = self.distilbert(
        #     input_ids=input_ids, attention_mask=attention_mask
        # )

        outputs = self.distilbert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        linear_output = self.dense1(
            outputs[0][:, 0, :]
        )  # outputs[0][:, 0, :] is the last hidden state, but is this right?

        # dropout_output = self.dropout(linear_output)
        relu_output = self.relu1(linear_output)
        output = self.classifier(relu_output)
        return self.sigmoid(output)
