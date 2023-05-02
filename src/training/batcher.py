import torch
import random

class Batcher():
    '''
    Allows for matched batching
    '''
    def __init__(self, data, tokenizer, bs=8, shuffle=True):
        self.shuffle = shuffle

        # Order by sequence length
        data = self._order(data)

        # Create batches
        batches = self._batch(data, bs)

        # Create token ids, pad by max batch size and create masks per batch
        self.batches = [self._batch_tokenize(b, tokenizer) for b in batches]

    
    def _order(self, data):
        return sorted(data, key=lambda d: len(d['text']))
    
    def _batch(self, data, bs):
        batches = [data[x:x+bs] for x in range(0, len(data), bs)]
        return batches
    
    def _batch_tokenize(self, batch, tokenizer):
        sentences = [b['text'] for b in batch]
        ml = tokenizer.model_max_length if tokenizer.model_max_length < 5000 else 512
        inputs = tokenizer(sentences, padding=True, max_length=ml, truncation=True, return_tensors="pt")
        # import pdb; pdb.set_trace()
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        y = torch.LongTensor([b['label'] for b in batch])
        return ids, mask, y
    
    # Called when iteration is initialized (for loop call this function is called in first line implicitly)
    def __iter__(self):
        self.i = 0
        if self.shuffle:
            random.shuffle(self.batches)
        return self
    
    # Called when moving to next element (called by for loop at beginning of each loop implicitly)
    def __next__(self):
        curr_i = self.i
        if curr_i > 0:
            # take previous tensors off device
            x,m,y = self.batches[curr_i-1]
            x=x.cpu(); m=m.cpu(); y=y.cpu()
            
        if curr_i >= len(self.batches):
            raise StopIteration
        
        self.i = curr_i + 1
        return self.batches[curr_i]
    
    def __len__(self):
        return len(self.batches)
    
        