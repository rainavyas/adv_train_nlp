from .model_selector import select_model
from ..training.trainer import Trainer

class Ensemble():
    def __init__(self, model_name, model_paths, device, num_labels=2):
        self.models = []
        for mpath in model_paths:
            model = select_model(model_name, mpath, num_labels=num_labels)
            model.to(device)
            self.models.append(model)

    def eval(self, dl, criterion, device):
        '''
        Evaluate Ensemble predictions
        Returns list of accuracies
        '''
        accs = []
        for m in self.models:
            acc = Trainer.eval(dl, m, criterion, device)
            accs.append(acc)
        return accs