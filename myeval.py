#%%
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from global_vars import normal_idx


# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

def compute_beta_score(labels, output, beta, num_classes, check_errors=True):

    # Check inputs for errors.
    if check_errors:
        if len(output) != len(labels):
            raise Exception('Numbers of outputs and labels must be the same.')

    # Populate contingency table.
    num_recordings = len(labels)

    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    # Weight function
    C_l=np.ones(num_classes);

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):
            
            num_labels = np.sum(labels[i])
        
            if labels[i][j] and output[i][j]:
                tp += 1/num_labels
            elif not labels[i][j] and output[i][j]:
                fp += 1/num_labels
            elif labels[i][j] and not output[i][j]:
                fn += 1/num_labels
            elif not labels[i][j] and not output[i][j]:
                tn += 1/num_labels

        # Summarize contingency table.
        if ((1+beta**2)*tp + (fn*beta**2) + fp):
            fbeta_l[j] = float((1+beta**2)* tp) / float(((1+beta**2)*tp) + (fn*beta**2) + fp)
        else:
            fbeta_l[j] = 1.0

        if (tp + fp + beta * fn):
            gbeta_l[j] = float(tp) / float(tp + fp + beta*fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0


    for i in range(num_classes):
        f_beta += fbeta_l[i]*C_l[i]
        g_beta += gbeta_l[i]*C_l[i]
        f_measure += fmeasure_l[i]*C_l[i]
        accuracy += accuracy_l[i]*C_l[i]


    f_beta = float(f_beta)/float(num_classes)
    g_beta = float(g_beta)/float(num_classes)
    f_measure = float(f_measure)/float(num_classes)
    accuracy = float(accuracy)/float(num_classes)


    return accuracy,f_measure,f_beta,g_beta

    
# The compute_auc function computes AUROC and AUPRC as well as other summary
# statistics (TP, FP, FN, TN, TPR, TNR, PPV, NPV, etc.) that can be exposed
# from this function.
#
# Inputs:
#   'labels' are the true classes of the recording
#
#   'output' are the output classes of your model
#
#   'beta' is the weight
#
#
# Outputs:
#   'auroc' is a scalar that gives the AUROC of the algorithm using its
#   output probabilities, where specificity is interpolated for intermediate
#   sensitivity values.
#
#   'auprc' is a scalar that gives the AUPRC of the algorithm using its
#   output probabilities, where precision is a piecewise constant function of
#   recall.
#



def compute_auc(labels, probabilities, num_classes, check_errors=True):


    # Check inputs for errors.
    if check_errors:
        if len(labels) != len(probabilities):
            raise Exception('Numbers of outputs and labels must be the same.')

    find_NaNs = np.isnan(probabilities);
    probabilities[find_NaNs] = 0;

    auroc_l = np.zeros(num_classes)
    auprc_l = np.zeros(num_classes)

    auroc = 0
    auprc = 0

    # Weight function - this will change
    C_l=np.ones(num_classes);

    # Populate contingency table.
    num_recordings = len(labels)

    for k in range(num_classes):
    

            # Find probabilities thresholds.
        thresholds = np.unique(probabilities[:,k])[::-1]
        if thresholds[0] != 1:
            thresholds = np.insert(thresholds, 0, 1)
        if thresholds[-1] == 0:
            thresholds = thresholds[:-1]

        m = len(thresholds)
    

        # Populate contingency table across probabilities thresholds.
        tp = np.zeros(m)
        fp = np.zeros(m)
        fn = np.zeros(m)
        tn = np.zeros(m)

        # Find indices that sort the predicted probabilities from largest to
        # smallest.
        idx = np.argsort(probabilities[:,k])[::-1]

        i = 0
        for j in range(m):
            # Initialize contingency table for j-th probabilities threshold.
            if j == 0:
                tp[j] = 0
                fp[j] = 0
                fn[j] = np.sum(labels[:,k])
                tn[j] = num_recordings - fn[j]
            else:
                tp[j] = tp[j - 1]
                fp[j] = fp[j - 1]
                fn[j] = fn[j - 1]
                tn[j] = tn[j - 1]
            # Update contingency table for i-th largest predicted probability.
            while i < num_recordings and probabilities[idx[i],k] >= thresholds[j]:
                if labels[idx[i],k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize contingency table.
        tpr = np.zeros(m)
        tnr = np.zeros(m)
        ppv = np.zeros(m)
        npv = np.zeros(m)


        for j in range(m):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = 1
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = 1
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = 1
            if fn[j] + tn[j]:
                npv[j] = float(tn[j]) / float(fn[j] + tn[j])
            else:
                npv[j] = 1

        # Compute AUROC as the area under a piecewise linear function with TPR /
        # sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR / recall (x-axis) and PPV / precision
        # (y-axis).

        for j in range(m-1):
            auroc_l[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc_l[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]


    for i in range(num_classes):
        auroc += auroc_l[i]*C_l[i]
        auprc += auprc_l[i]*C_l[i]

    auroc = float(auroc)/float(num_classes)
    auprc = float(auprc)/float(num_classes)

    
    return auroc, auprc



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


def binary_acc_core(y_test_numpy, y_pred_prob_numpy):
    # auroc

    auroc = roc_auc_score(y_test_numpy, y_pred_prob_numpy)

    # auprc
    precision, recall, thresholds = precision_recall_curve(y_test_numpy, y_pred_prob_numpy)
    auprc = auc(recall, precision)

    # binary result
    return auroc, auprc

def agg_y_preds(outputs):
    y_pred_probs = torch.sigmoid(outputs)
    y_pred_prob_max, _ = torch.max(y_pred_probs, axis=0)
    y_pred_prob_mean = torch.mean(y_pred_probs, axis=0)
    
    return y_pred_prob_max, y_pred_prob_mean

def agg_y_preds_bags(ys, bag_size):
    n_bags = int(len(ys)/bag_size)
    ys_bags_mean = [torch.mean(ys[i*bag_size:i*bag_size+bag_size], axis=0) for i in range(n_bags)]
    ys_bags_max = [torch.max(ys[i*bag_size:i*bag_size+bag_size], axis=0)[0] for i in range(n_bags)]
    ys_bags_first = [ys[i*bag_size] for i in range(n_bags)]
    
    return torch.stack(ys_bags_max, axis=0), torch.stack(ys_bags_mean, axis=0), torch.stack(ys_bags_first, axis=0)
    
def binary_acc_mic(y_preds, y_tests, beta=2):
    accs = []
    fmeasures = []
    fbetas = []
    gbetas = []
    aurocs = []
    auprcs = []
    for i in range(9):
        y_pred, y_test = y_preds[:,i], y_tests[:,i]
        
        # prob
        if 'cuda'  in y_pred.device.type:
            y_test_numpy = y_test.data.cpu().numpy()
            y_pred_prob_numpy = y_pred.data.cpu().numpy()
        else:
            y_test_numpy = y_test.data.numpy()
            y_pred_prob_numpy = y_pred.data.numpy()
    
        auroc, auprc = binary_acc_core(y_test_numpy, y_pred_prob_numpy)
        # old way to cal acc:
        #correct_results_sum = (y_pred_tag == y_test).sum().float()
        #acc = true_positives/y_test.shape[0]
        #acc = torch.round(acc * 100)
        
        y_pred_tag = torch.round(y_pred_prob)
        tp, fp, tn, fn = confusion(y_pred_tag, y_test)

        # acc, fmeasure, fbeta, gbeta
        acc = float(tp + tn) / float(tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 1.0
        fmeasure = float(2 * tp) / float(2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
        fbeta = float((1+beta**2)* tp) / float(((1+beta**2)*tp) + (fn*beta**2) + fp) if ((1+beta**2)*tp) + (fn*beta**2) + fp > 0 else 1.0
        gbeta = float(tp) / float(tp + fp + beta*fn) if tp + fp + beta*fn > 0 else 1.0
    

        accs.append(acc)#.data.cpu().numpy())
        fbetas.append(fbeta)#.data.cpu().numpy())
        fmeasures.append(fmeasure)
        gbetas.append(gbeta)
        aurocs.append(auroc)
        auprcs.append(auprc)
    #return accs, fbetas, fmeasures, gbetas, aurocs, auprcs
    return np.mean(accs), np.mean(fbetas), np.mean(fmeasures), np.mean(gbetas), np.mean(aurocs), np.mean(auprcs)

def binary_acc(y_preds, y_tests, beta=1, mode='mean'):
    accs = []
    fmeasures = []
    gmeasures = []
    fbetas = []
    gbetas = []
    aurocs = []
    auprcs = []
    for i in range(y_preds.shape[1]):
        
        # Tensor
        y_pred_prob, y_test = y_preds[:,i], y_tests[:,i]
        
        y_pred_tag = torch.round(y_pred_prob)
        tp, fp, tn, fn = confusion(y_pred_tag, y_test)
        
        
        # numpy array 
        y_test_numpy = None
        y_pred_prob_numpy = None
        
        if 'cuda'  in y_pred_prob.device.type:
            y_test_numpy = y_test.data.cpu().numpy()
            y_pred_prob_numpy = y_pred_prob.data.cpu().numpy()
        else:
            y_test_numpy = y_test.data.numpy()
            y_pred_prob_numpy = y_pred_prob.data.numpy()
    
  #      auroc, auprc = binary_acc_core(y_test_numpy, y_pred_prob_numpy)
        # old way to cal acc:
        #correct_results_sum = (y_pred_tag == y_test).sum().float()
        #acc = true_positives/y_test.shape[0]
        #acc = torch.round(acc * 100)
        

        # acc, fmeasure, fbeta, gbeta
        acc = float(tp + tn) / float(tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 1.0
        fmeasure = float(2 * tp) / float(2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
        gmeasure = float(tp) / float(tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        fbeta = float((1+beta**2)* tp) / float(((1+beta**2)*tp) + (fn*beta**2) + fp) if ((1+beta**2)*tp) + (fn*beta**2) + fp > 0 else 1.0
        gbeta = float(tp) / float(tp + fp + beta*fn) if tp + fp + beta*fn > 0 else 1.0
    

        accs.append(acc)#.data.cpu().numpy())
        fbetas.append(fbeta)#.data.cpu().numpy())
        fmeasures.append(fmeasure)
        gmeasures.append(gmeasure)
        gbetas.append(gbeta)
 #       aurocs.append(auroc)
 #       auprcs.append(auprc)
    
    if mode == 'mean':
        return np.mean(accs), np.mean(fmeasures), np.mean(gmeasures), np.mean(fbetas), np.mean(gbetas)#, np.mean(aurocs), np.mean(auprcs)
    return accs, fmeasures, gmeasures, fbetas, gbetas#, aurocs, auprcs

def geometry_loss(fbeta, gbeta):
    return np.sqrt(fbeta*gbeta)

def compute_score(y_labels, y_outputs, weights, class_idx=list(range(27)), normal_index=normal_idx):
    # use a subset of class
    weights = weights[class_idx, class_idx]

    num_recordings, num_classes = np.shape(y_labels)
    # Compute the observed score.
    A = compute_modified_confusion_matrix(y_labels, y_outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = y_labels
    A = compute_modified_confusion_matrix(y_labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the normal class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(y_labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = float('nan')

    return normalized_score