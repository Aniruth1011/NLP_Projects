import torch
from transformers import BertTokenizer, BertForSequenceClassification

model_name = 'bert-base-uncased'  
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

text = "I really enjoyed this movie! It was fantastic."

inputs = tokenizer(text, return_tensors='pt')


outputs = model(**inputs)

predictions = torch.softmax(outputs.logits, dim=1)
predicted_class = torch.argmax(predictions, dim=1).item()
predicted_label = model.config.id2label[predicted_class]

print(f"Predicted class: {predicted_class} - Predicted label: {predicted_label}")
