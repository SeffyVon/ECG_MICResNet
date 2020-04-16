#%%
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from evaluate_12ECG_score import compute_beta_score, compute_auc

# def evaluate_beta(output, y):
    
#     accuracy,f_measure,f_beta,g_beta = compute_beta_score(labels=y, 
#                        output=output, 
#                        beta=2, num_classes=1)
    
#     auroc, auprc = compute_auc(labels=y, 
#                                 probabilities=output,
#                                 num_classes=1)

#     return accuracy,f_measure,f_beta,g_beta, auroc, auprc


def confusion(prediction, truth):
    
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

def binary_acc(y_preds, y_tests, beta=2):
    accs = []
    fmeasures = []
    fbetas = []
    gbetas = []
    aurocs = []
    auprcs = []
    for i in range(9):
        y_pred, y_test = y_preds[:,i], y_tests[:,i]
        
        # prob
        y_pred_prob = torch.sigmoid(y_pred)
        
        # auroc
        y_test_numpy = y_test.data.cpu().numpy()
        y_pred_prob_numpy = y_pred_prob.data.cpu().numpy()
        auroc = roc_auc_score(y_test_numpy, y_pred_prob_numpy)
        
        # auprc
        precision, recall, thresholds = precision_recall_curve(y_test_numpy, y_pred_prob_numpy)
        auprc = auc(recall, precision)

        # binary result
        y_pred_tag = torch.round(y_pred_prob)
        tp, fp, tn, fn = confusion(y_pred_tag, y_test)
        
        # acc, fmeasure, fbeta, gbeta
        acc = float(tp + tn) / float(tp + fp + fn + tn)
        fmeasure = float(2 * tp) / float(2 * tp + fp + fn)
        fbeta = float((1+beta**2)* tp) / float(((1+beta**2)*tp) + (fn*beta**2) + fp)
        gbeta = float(tp) / float(tp + fp + beta*fn)
        
        # old way to cal acc:
        #correct_results_sum = (y_pred_tag == y_test).sum().float()
        #acc = true_positives/y_test.shape[0]
        #acc = torch.round(acc * 100)

        accs.append(acc)#.data.cpu().numpy())
        fbetas.append(fbeta)#.data.cpu().numpy())
        fmeasures.append(fmeasure)
        gbetas.append(gbeta)
        aurocs.append(auroc)
        auprcs.append(auprc)
    return np.mean(accs), np.mean(fbetas), np.mean(fmeasures), np.mean(gbetas), np.mean(aurocs), np.mean(auprcs)