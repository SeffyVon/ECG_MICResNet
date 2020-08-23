
from manipulations import get_scored_class, get_name, cv_split
from global_vars import labels, equivalent_mapping, Dx_map, Dx_map_unscored, \
    normal_class, weights, disable_tqdm, enable_writer, run_name_base, n_segments, max_segment_len
from resnet1d import ECGBagResNet
from dataset import BagSigDataset
from myeval import agg_y_preds_bags, binary_acc, geometry_loss, compute_score
from imbalanced_weights import inverse_weight
from pytorchtools import EarlyStopping, add_pr_curve_tensorboard
from saved_data_io import read_file 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
if enable_writer:
    from torch.utils.tensorboard import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


def train_NN_sig_MIL(headers_datasets, output_directory, fDatas):

    for i_fold in [2, 3, 4, 5, 6]:

        run_name = run_name_base + '_' +  str(i_fold)

        Codes, dataset_train_idx, dataset_test_idx, filenames = cv_split(headers_datasets, i_fold=i_fold)

        
        datasets = np.sort(list(headers_datasets.keys()))

        # agg labels
        data_img2_labels = []
        for i in tqdm(range(len(Codes)), disable=disable_tqdm):
            data_img2_labels.append(get_scored_class(Codes[i], labels))
        data_img2_labels = np.array(data_img2_labels)
        assert len(data_img2_labels) == len(Codes)

        # change to equivalent mapping
        for key in equivalent_mapping.keys():
            print('equivalent', key, equivalent_mapping[key])
            key_idx = np.argwhere(labels==int(key)).flatten()[0]
            val_idx = np.argwhere(labels==int(equivalent_mapping[key])).flatten()[0]
            key_pos = np.argwhere(data_img2_labels[:,key_idx]==1).flatten()
            val_pos = np.argwhere(data_img2_labels[:,val_idx]==1).flatten()
            data_img2_labels[key_pos,val_idx] = 1
            data_img2_labels[val_pos,key_idx] = 1

        # agg CV split
        train_idx = []
        test_idx = []
        for dataset in datasets:
            for idx in dataset_train_idx[dataset]:
                train_idx.append(idx)
            for idx in dataset_test_idx[dataset]:
                test_idx.append(idx)

        print('CV split:', i_fold)
        print(np.sort(train_idx)[:5], np.sort(test_idx)[:5])

        continue

        assert len(np.unique(train_idx))+len(np.unique(test_idx)) == len(Codes)
        assert len(np.unique(train_idx+test_idx)) == len(Codes)
        print('CV split checked')

        del Codes, dataset_train_idx, dataset_test_idx, headers_datasets

        names = [get_name(label, Dx_map, Dx_map_unscored) for label in labels]
        class_idx = np.argwhere(np.sum(np.array(data_img2_labels)[train_idx],axis=0)!=0).flatten() 
        names = np.array(names)[class_idx]
        normal_idx = np.argwhere(labels[class_idx]==int(normal_class)).flatten()[0]

        print("#classes: ", len(class_idx), "data_img2_labels.dim", data_img2_labels.shape)
        print("normal_idx: ", normal_idx)

        # get device
        device = None
        if torch.cuda.is_available():
            device = torch.device('cuda:1')
        else:
            device = torch.device('cpu')

        torch.manual_seed(0)

        st = time.time()

        train_class_weight = torch.Tensor(inverse_weight(data_img2_labels[train_idx], class_idx)).to(device)
        test_class_weight = torch.Tensor(inverse_weight(data_img2_labels[test_idx], class_idx)).to(device)
        print("train_class_weight", train_class_weight)
        print("test_class_weight", test_class_weight)

        sig_datasets_train = BagSigDataset(fDatas, data_img2_labels, 
            class_idx, 'train', n_segments, max_segment_len)
        sig_datasets_test = BagSigDataset(fDatas, data_img2_labels, 
            class_idx, 'test', n_segments, max_segment_len)

        trainDataset = torch.utils.data.Subset(sig_datasets_train, train_idx)
        testDataset = torch.utils.data.Subset(sig_datasets_test, test_idx)

        batch_size = 64
        trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=0)
           # sampler=ImbalancedDatasetSampler(trainDataset, callback_get_label=lambda x, i: tuple(x[i][1].tolist())))
        testLoader = torch.utils.data.DataLoader(testDataset, batch_size=300, shuffle=False, pin_memory=True, num_workers=0)
        pos_weight = torch.from_numpy(np.array([2 for _ in range(len(class_idx))])).to(device)

        criterion_train = nn.BCEWithLogitsLoss(reduction='mean', weight=train_class_weight, pos_weight=pos_weight)
        criterion_test = nn.BCEWithLogitsLoss(reduction='mean', weight=test_class_weight, pos_weight=pos_weight)
        early_stopping = EarlyStopping(patience=20, verbose=False, 
                                      saved_dir=output_directory, 
                                      save_name=run_name)


        model = ECGBagResNet(12, len(class_idx), n_segments)
        with open(output_directory + '/ECGBagResNet' + run_name + '.txt', 'w') as f:
           print(model, file=f)

        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.01) 
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, mode='min')

        losses_train = []
        losses_test = []
        
        if enable_writer:
            writer = SummaryWriter(output_directory+'/runs/{}'.format(run_name))


        # training
        for epoch in range(0, 100):

            model.train()

            y_trains = [] # ground truth
            output_trains = [] # output
            with tqdm(enumerate(trainLoader), desc='train', disable=disable_tqdm) as tqdm0:
                for k, (X_sig_train, y_train) in tqdm0:
                    X_sig_train = X_sig_train.to(device)
                    y_train = y_train.to(device)

                    optimizer.zero_grad()
                    output_train = model(X_sig_train)
                    output_trains.append(output_train.cpu())
                    
                    loss_train = criterion_train(output_train, y_train)
                    losses_train.append(loss_train.item())
                    
                    avg_loss_train = np.average(losses_train)

                    if enable_writer:
                        if np.mod(k, 100) == 0:
                            writer.add_scalar('train/loss',
                            avg_loss_train,
                            epoch * (len(train_idx)//batch_size//100+1) + k//100)

                    y_trains.append(y_train.cpu())
                    tqdm0.set_postfix(loss=avg_loss_train)

                    loss_train.backward()
                    optimizer.step()
                    
            y_tests = [] # ground truth
            output_tests = [] # output
            with torch.no_grad():
                model.eval()
                
                for X_sig_test, y_test in tqdm(testLoader, desc='test', disable=disable_tqdm):  
                    y_test = y_test.to(device)
                    X_sig_test = X_sig_test.to(device)
                    output_test = model(X_sig_test)

                    loss_test = criterion_test(output_test, y_test)
                    losses_test.append(loss_test.item())

                    output_tests.append(output_test.cpu())
                    y_tests.append(y_test.cpu())
                    
                avg_loss_test = np.average(losses_test)

                if enable_writer:
                    writer.add_scalar('test/loss',
                        avg_loss_test,
                        epoch)


            y_trains_tensor = torch.cat(y_trains, axis=0) # ground truth
            y_tests_tensor = torch.cat(y_tests, axis=0) # ground truth

            output_trains = torch.cat(output_trains, axis=0) 
            y_train_preds = torch.sigmoid(output_trains)

            output_tests = torch.cat(output_tests, axis=0)
            y_test_preds = torch.sigmoid(output_tests)

            if enable_writer:
                for class_i_idx in range(len(class_idx)):
                    add_pr_curve_tensorboard(writer, class_i_idx, y_trains_tensor, y_train_preds, names, global_step=epoch, prefix='train/')
                    add_pr_curve_tensorboard(writer, class_i_idx, y_tests_tensor, y_test_preds, names, global_step=epoch, prefix='test/')



            acc, fmeasure, gmeasure, fbeta, gbeta = binary_acc(y_train_preds, y_trains_tensor)           
            acc2, fmeasure2, gmeasure2, fbeta2, gbeta2 = binary_acc(y_test_preds, y_tests_tensor)

            geometry = geometry_loss(fbeta, gbeta)
            geometry2 = geometry_loss(fbeta2, gbeta2)
            geometry_1 = geometry_loss(fmeasure, gmeasure)
            geometry2_1 = geometry_loss(fmeasure2, gmeasure2)

            
            score = compute_score(np.round(y_train_preds.data.numpy()), np.round(y_trains_tensor.data.numpy()),  weights, class_idx, normal_idx)
            score2 = compute_score(np.round(y_test_preds.data.numpy()), np.round(y_tests_tensor.data.numpy()), weights, class_idx, normal_idx)
            output_str = 'S{}/{} {:.2f} min |\n Train Loss: {:.6f}, Acc: {:.3f}, F: {:.3f}, G: {:.3f}, Fbeta: {:.3f}, gbeta: {:.3f}, geo_1: {:.3f}, geo: {:.3f}, score: {:.3f} |\n Valid Loss: {:.6f}, Acc: {:.3f}, F: {:.3f}, G: {:.3f}, Fbeta: {:.3f}, gbeta: {:.3f}, geo_1: {:.3f}, geo: {:.3f}, score: {:.3f}\n '.format(
                i_fold, epoch, (time.time()-st)/60,
                avg_loss_train, acc, fmeasure, gmeasure, fbeta, gbeta, geometry_1, geometry, score,
                avg_loss_test, acc2, fmeasure2, gmeasure2, fbeta2, gbeta2, geometry2_1, geometry2, score2)
            scheduler.step(avg_loss_test)

            if enable_writer:
                writer.add_scalar('train/score',
                    score,
                    epoch)
                writer.add_scalar('train/fmeasure',
                        fmeasure,
                        epoch)
                writer.add_scalar('train/gmeasure',
                        gmeasure,
                        epoch)
                writer.add_scalar('train/geometry_1',
                        geometry_1,
                        epoch)
                writer.add_scalar('train/gbeta',
                        gbeta,
                        epoch)
                writer.add_scalar('train/fbeta',
                        fbeta,
                        epoch)
                writer.add_scalar('train/geometry',
                        geometry,
                        epoch)


                writer.add_scalar('test/score',
                        score2,
                        epoch)
                writer.add_scalar('test/fmeasure',
                        fmeasure2,
                        epoch)
                writer.add_scalar('test/gmeasure',
                        gmeasure2,
                        epoch)
                writer.add_scalar('test/geometry_1',
                        geometry2_1,
                        epoch)
                writer.add_scalar('test/gbeta',
                        gbeta2,
                        epoch)
                writer.add_scalar('test/fbeta',
                        fbeta2,
                        epoch)
                writer.add_scalar('test/geometry',
                        geometry2,
                        epoch)


            
            print(output_str)        
            with open(output_directory+'/loss_{}.txt'.format(run_name), 'a') as f:
                print(output_str, file=f)

            early_stopping(avg_loss_test, model, epoch)

            if early_stopping.early_stop:
                print("Early stopping with min validation loss:", early_stopping.val_loss_min)
                break
