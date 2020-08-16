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

