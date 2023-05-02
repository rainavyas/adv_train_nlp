from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch.nn as nn
import torch

class SequenceClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2, pretrained=True):
        super().__init__()
        self.model_name = model_name
        if pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels) # returns config and not pretrained weights 
            self.model = AutoModelForSequenceClassification.from_config(config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)[0]
    
    def predict(self, sentences, output_attentions=False, output_hidden_states=False, return_dict=False, device=torch.device('cpu')):
        ml = self.tokenizer.model_max_length if self.tokenizer.model_max_length < 5000 else 512
        inputs = self.tokenizer(sentences, padding=True, max_length=ml, truncation=True, return_tensors="pt")
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        return(self.model(input_ids, attention_mask, output_attentions=output_attentions,
                 output_hidden_states=output_hidden_states, return_dict=return_dict))



        