import pickle
import numpy as np

data_path_dict = {
    'Q_locs': '../saved/newData_Q_locs_1000_7000_peakdist100.pkl',
    'Codes': '../saved/newData_Codes_1000_7000_peakdist100.pkl',
    'dataset_train_idx': '../saved/dataset_train_idx.pkl',
    'dataset_test_idx': '../saved/dataset_test_idx.pkl',
    'Data_labels_train': '../saved/Data_labels_train_stratified.npy',
    'Data_labels_test': '../saved/Data_labels_test_stratified.npy',
    'Signals_train': '../saved/Signals_train_stratified.npy',
    'Signals_test': '../saved/Signals_test_stratified.npy' ,
    'data_imgs_dataset1' :'../saved/data_imgs2.pkl',
    'data_imgs_dataset2' :'../saved/data_imgs_dataset2.pkl',
    'data_imgs_dataset2_2' :'../saved/data_imgs_dataset2_2.pkl',
    'data_imgs_dataset2_new' :'../saved/data_imgs_dataset2_new.pkl',
}
def read_file(name, default_saved_path=False, verbose=False):
    res = None
    saved_path = None

    if name[-4:] in ['.npy', '.pkl']:
        saved_path = name
    else:
        if not default_saved_path:
            saved_path = data_path_dict[name]
        else:
            saved_path = 'saved/{}.pkl'.format(name)

    if saved_path[-4:] == '.npy':
        res = np.load(saved_path)
    else:
        with open(saved_path, 'rb') as saved_file:
            res = pickle.load(saved_file)

    if verbose:
        print('read from', saved_path)
    return res

def write_file(name, obj, default_saved_path=False, verbose=False):

    saved_path = None
    if name[-4:] in ['.npy', '.pkl']:
        saved_path = name
    else:
        if not default_saved_path:
            saved_path = data_path_dict[name]
        else:
            saved_path = 'saved/{}.pkl'.format(name)

    if saved_path[-4:] == '.npy':
        np.save(saved_path, obj)

    else:
        with open(saved_path, 'wb') as saved_file:
            pickle.dump(obj, saved_file)
    if verbose:
        print('saved at', saved_path)