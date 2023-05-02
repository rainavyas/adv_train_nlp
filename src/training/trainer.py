from torch.utils.data import TensorDataset, DataLoader
import torch 

from ..tools.tools import AverageMeter, accuracy_topk, print_log
from .regularization import loss_reg


class Trainer():
    '''
    All training functionality
    '''
    def __init__(self, device, model, optimizer, criterion, scheduler):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
    
    @staticmethod
    def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25, reg=None, reg_beta=1.0):
        '''
        Run one train epoch
        '''
        losses = AverageMeter()
        accs = AverageMeter()

        # switch to train mode
        model.train()

        for i, (ids, mask, y) in enumerate(train_loader):

            ids = ids.to(device)
            mask = mask.to(device)
            y = y.to(device)
            # Forward pass
            logits = model(ids, mask)
            loss = criterion(logits, y)
            if reg is not None: loss += reg_beta*loss_reg(reg, model)

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, y)
            accs.update(acc.item(), ids.size(0))
            losses.update(loss.item(), ids.size(0))

            if i % print_freq == 0:
                print_log(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})\tAccuracy {accs.val:.3f} ({accs.avg:.3f})')


    @staticmethod
    def eval(val_loader, model, criterion, device, return_logits=False):
        '''
        Run evaluation
        '''
        losses = AverageMeter()
        accs = AverageMeter()

        # switch to eval mode
        model.eval()

        all_logits = []
        with torch.no_grad():
            for (ids, mask, y) in val_loader:

                ids = ids.to(device)
                mask = mask.to(device)
                y = y.to(device)

                # Forward pass
                logits = model(ids, mask)
                all_logits.append(logits)
                loss = criterion(logits, y)

                # measure accuracy and record loss
                acc = accuracy_topk(logits.data, y)
                accs.update(acc.item(), ids.size(0))
                losses.update(loss.item(), ids.size(0))

        if return_logits:
            return torch.cat(all_logits, dim=0).detach().cpu()

        print_log(f'Test\t Loss ({losses.avg:.4f})\tAccuracy ({accs.avg:.3f})\n')
        return accs.avg

    @staticmethod
    def prep_dl(model, data, bs=8, shuffle=True):
        '''
            Prep data into dataloader
        '''
        # Get ids and mask
        sentences = [d['text'] for d in data]
        inputs = model.tokenizer(sentences, padding=True, max_length=model.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        y = torch.LongTensor([d['label'] for d in data])

        ds = TensorDataset(ids, mask, y)
        dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
        return dl
    
    def train_process(self, train_data, val_data, save_path, max_epochs=10, bs=8, reg=None, reg_beta=1.0):

        train_dl = self.prep_dl(self.model, train_data, bs=bs, shuffle=True)
        val_dl = self.prep_dl(self.model, val_data, bs=bs, shuffle=True)

        best_acc = 0
        for epoch in range(max_epochs):
            # train for one epoch
            print_log('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.train(train_dl, self.model, self.criterion, self.optimizer, epoch, self.device, reg=reg, reg_beta=reg_beta)
            self.scheduler.step()

            # evaluate on validation set
            acc = self.eval(val_dl, self.model, self.criterion, self.device)
            if acc > best_acc:
                best_acc = acc
                state = self.model.state_dict()
                torch.save(state, save_path)
            