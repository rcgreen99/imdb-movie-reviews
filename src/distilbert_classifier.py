from transformers import DistilBertModel, DistilBertForSequenceClassification
from torch import nn

# create custom model
class DistilBertClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.distilbert_model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        # for param in self.distilbert_model.parameters():
        #     param.requires_grad = False

        # self.dense1 = nn.Linear(768, 768)
        # self.relu1 = nn.ReLU()
        # self.dropout_1 = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # _, pooled_output = self.distilbert(
        #     input_ids=input_ids, attention_mask=attention_mask
        # )

        outputs = self.distilbert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # dropout_1 = self.dropout_1(outputs[0][:, 0, :])
        # linear_output = self.dense1(outputs[0][:, 0, :])
        # relu_output = self.relu1(linear_output)
        # dropout_1 = self.dropout_1(relu_output)
        output = self.classifier(outputs[0][:, 0, :])
        return self.sigmoid(output)
