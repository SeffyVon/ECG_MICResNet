import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, saved_dir='.', save_name=''):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_name = save_name
        self.saved_dir = saved_dir

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if np.mod(epoch, 10) == 0:
            self.save_checkpoint(val_loss, model, epoch)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0


    def save_checkpoint(self, val_loss, model, epoch='best'):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), '{}/{}_model_{}.dict'.format(self.saved_dir, self.save_name, epoch))
        #torch.save(model, '{}/{}_checkpoint.pt'.format(self.saved_dir, self.save_name))
        self.val_loss_min = val_loss

def add_pr_curve_tensorboard(writer, class_index, test_probs, test_preds, names, global_step, prefix):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''

    writer.add_pr_curve(prefix + '_' + names[class_index],
                        test_preds[:,class_index],
                        test_probs[:,class_index],
                        global_step=global_step,
                        num_thresholds=127)
    writer.close()


