from .trainer import Trainer
from .batcher import Batcher

class BatchTrainer(Trainer):
    '''
    All training functionality with matched batching
        Matched batching does the following:
            1) Batches similar length sequences
            2) Pads to maximum length of batch as opposed to entire dataset
    '''
    def __init__(self, device, model, optimizer, criterion, scheduler):
        super().__init__(device, model, optimizer, criterion, scheduler)

    @staticmethod
    def prep_dl(model, data, bs=8, shuffle=True):
        '''
            Prep data into dataloader
            Use matched bathching with manual batcher
        '''
        # Get ids and mask using matched batcher
        dl = Batcher(data, model.tokenizer, bs=bs, shuffle=shuffle)
        return dl