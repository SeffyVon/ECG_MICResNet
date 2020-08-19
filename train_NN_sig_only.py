
from manipulations import get_classes_from_header, get_scored_class, get_name
from global_vars import labels, equivalent_mapping, Dx_map, Dx_map_unscored, equivalent_mapping, \
    normal_class, weights, disable_tqdm, enable_writer
from resnet1d import ECGResNet
from IdvImageSigDataset import IdvSigDataset
from myeval import agg_y_preds_bags, binary_acc, geometry_loss, compute_score

from pytorchtools import EarlyStopping
from pytorch_training import add_pr_curve_tensorboard
from saved_data_io import read_file 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch, torchvision 
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
if enable_writer:
    from torch.utils.tensorboard import SummaryWriter
import time
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


def cv_split(headers_datasets):
    """
    80-20 stratified CV split across each dataset
    """
    
    Codes = []
    
    dataset_idx = {}
    dataset_data_labels = {} # encoding
    dataset_train_idx = {}
    dataset_test_idx = {}
    
    datasets = np.sort(list(headers_datasets.keys()))
    filenames = []
    global_idx = 0
    for dataset in datasets:
        print('Dataset ', dataset)
        headers_dataset = headers_datasets[dataset]
        num_files = len(headers_dataset)
        dataset_idx[dataset] = []
        dataset_data_labels[dataset] = []
        for i, header_data in tqdm(enumerate(headers_dataset), disable=disable_tqdm):
            
            codes = get_classes_from_header(header_data)
            filename = header_data[0].split(' ')[0]
            data_labels = get_scored_class(codes, labels)

            Codes.append(codes)
            filenames.append(filename)
            
            dataset_data_labels[dataset].append(data_labels)
            dataset_idx[dataset].append(global_idx)
            global_idx += 1
        
        kf = MultilabelStratifiedKFold(5, random_state=0)
        train_idx, test_idx = next(kf.split(np.array(dataset_data_labels[dataset]), np.array(dataset_data_labels[dataset])))
        dataset_train_idx[dataset] = train_idx +  dataset_idx[dataset][0]
        dataset_test_idx[dataset] = test_idx + dataset_idx[dataset][0]
        
        print('Done.')
    return Codes, dataset_train_idx, dataset_test_idx, filenames

def get_dataset(headers, recordings=None):

    dataset_mapping = {
        'A': 1,
        'Q': 2,
        'I': 3,
        'S': 4,
        'H': 5,
        'E': 6
    }
    if recordings is not None:
        headers_datasets = {}
        recordings_datasets = {}
        for i, (header, recording) in enumerate(zip(headers, recordings)):
            dataset = dataset_mapping[header[0].split(' ')[0][0]]
            if dataset in headers_datasets:
                headers_datasets[dataset].append(header)
                recordings_datasets[dataset].append(recording)
            else:
                headers_datasets[dataset] = [header]
                recordings_datasets[dataset] = [recording]
        return headers_datasets, recordings_datasets

    else:
        headers_datasets = {}
        for i, header in enumerate(headers):
            dataset = dataset_mapping[header[0].split(' ')[0][0]]
            if dataset in headers_datasets:
                headers_datasets[dataset].append(header)
            else:
                headers_datasets[dataset] = [header]
        return headers_datasets        


def train_NN_sig_only(headers_datasets, output_directory):

    Codes, dataset_train_idx, dataset_test_idx, filenames = cv_split(headers_datasets)


    datasets = np.sort(list(headers_datasets.keys()))

    # agg labels
    data_img2_labels = []
    for i in tqdm(range(len(Codes)), disable=disable_tqdm):
        data_img2_labels.append(get_scored_class(Codes[i], labels))
    data_img2_labels = np.array(data_img2_labels)
    assert len(data_img2_labels) == len(Codes)

    # change to equivalent mapping
    key_idxes = []
    for key in equivalent_mapping.keys():
        print(key)
        key_idx = np.argwhere(labels==int(key)).flatten()[0]
        key_idxes.append(key_idx)
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

    assert len(train_idx)+len(test_idx) == len(Codes)

    del Codes, dataset_train_idx, dataset_test_idx, headers_datasets

    names = [get_name(label, Dx_map, Dx_map_unscored) for label in labels]
    class_idx = np.argwhere(np.sum(np.array(data_img2_labels)[train_idx],axis=0)!=0).flatten() 
    names = np.array(names)[class_idx]
    normal_idx = np.argwhere(labels[class_idx]==int(normal_class)).flatten()[0]

    print("#classes: ", len(class_idx))
    print("normal_idx: ", normal_idx)

    # get device
    device = None
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    torch.manual_seed(0)

    st = time.time()

   # train_class_weight = torch.Tensor(inverse_weight(data_img2_labels[train_idx], class_idx)).to(device)
   # test_class_weight = torch.Tensor(inverse_weight(data_img2_labels[test_idx], class_idx)).to(device)

    sig_datasets_train = IdvSigDataset(output_directory, filenames, data_img2_labels, 
        class_idx, 'train')
    sig_datasets_test = IdvSigDataset(output_directory, filenames, data_img2_labels, 
        class_idx, 'test')

    trainDataset = torch.utils.data.Subset(sig_datasets_train, train_idx)
    testDataset = torch.utils.data.Subset(sig_datasets_test, test_idx)

    batch_size = 64
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, pin_memory=True, shuffle=True,
                                              num_workers=16)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=300, shuffle = False, pin_memory=True,
                                              num_workers=16)


    criterion_train = nn.BCEWithLogitsLoss(reduction='mean')#, weight=train_class_weight)
    criterion_test = nn.BCEWithLogitsLoss(reduction='mean')#, weight=test_class_weight)

    run_name = 'modelMultiCWTFull_test_sigOnly'

    early_stopping = EarlyStopping(patience=50, verbose=False, 
                                  saved_dir=output_directory, 
                                  save_name=run_name)

    model = ECGResNet(12, len(class_idx))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01) 
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, mode='min')

    losses_train = []
    losses_test = []
    
    if enable_writer:
        writer = SummaryWriter(output_directory+'/runs/{}'.format(run_name))


    # training
    for epoch in range(0, 500):

        model.train()

        y_trains = [] # ground truth
        output_trains = [] # output
        for k, (X_sig_train, y_train) in tqdm(enumerate(trainLoader), desc='train', disable=disable_tqdm):
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



        acc, fmeasure, fbeta, gbeta = binary_acc(y_train_preds, y_trains_tensor)           
        acc2, fmeasure2, fbeta2, gbeta2 = binary_acc(y_test_preds, y_tests_tensor)
        geometry = geometry_loss(fbeta, gbeta)
        geometry2 = geometry_loss(fbeta2, gbeta2)
        
        score = compute_score(np.round(y_train_preds.data.numpy()), np.round(y_trains_tensor.data.numpy()),  weights, class_idx, normal_idx)
        score2 = compute_score(np.round(y_test_preds.data.numpy()), np.round(y_tests_tensor.data.numpy()), weights, class_idx, normal_idx)
        output_str = 'S{} {:.2f} min |\n Train Loss: {:.6f}, Acc: {:.3f}, F: {:.3f}, Fbeta: {:.3f}, gbeta: {:.3f}, geo: {:.3f}, score: {:.3f} |\n Valid Loss: {:.6f}, Acc: {:.3f}, F: {:.3f}, Fbeta: {:.3f}, gbeta: {:.3f}, geo: {:.3f}, score: {:.3f}\n '.format(
            epoch, (time.time()-st)/60,
            avg_loss_train, acc, fmeasure, fbeta, gbeta, geometry, score,
            avg_loss_test, acc2, fmeasure2, fbeta2, gbeta2, geometry2, score2)
        scheduler.step(avg_loss_test)

        if enable_writer:
            writer.add_scalar('train/score',
                score,
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

        early_stopping(avg_loss_test, model)

        if early_stopping.early_stop:
            print("Early stopping with min validation loss:", early_stopping.val_loss_min)
            break
