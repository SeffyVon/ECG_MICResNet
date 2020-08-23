
import numpy as np, os, sys
from scipy.io import loadmat
from tqdm.notebook import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from global_vars import labels, normal_class, equivalent_mapping, disable_tqdm

def load_challenge_data(filename):


    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data

def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')

  
# Find unique number of classes  
def get_classes(input_directory,files):

    classes=[]
    for f in files:
        g = f.replace('.mat','.hea')
        input_file = os.path.join(input_directory,g)
        with open(input_file,'r') as f:
            classes += get_classes_from_header(f)

    return sorted(set(classes))

def get_classes_from_header(header_data):
    classes = []
    for lines in header_data:
        if lines.startswith('#Dx'):
            tmp = lines.split(': ')[1].split(',')
            for c in tmp:
                classes.append(int(c.strip()))
    return sorted(classes)

def get_Fs_from_header(header_data):
    fst_line = header_data[0].split(' ')
    return int(fst_line[2])

def get_abbr(code, Dx_map, Dx_map_unscored):
    entry = None
    if code in list(Dx_map['SNOMED CT Code']):
        entry = list(Dx_map[Dx_map['SNOMED CT Code']==code]['Abbreviation'])[0]
    else:
        entry = '*'+list(Dx_map_unscored[Dx_map_unscored['SNOMED CT Code']==code]['Abbreviation'])[0]
    return entry

def get_name(code, Dx_map, Dx_map_unscored):
    entry = None
    if code in list(Dx_map['SNOMED CT Code']):
        entry = list(Dx_map[Dx_map['SNOMED CT Code']==code]['Dx'])[0]
    else:
        entry = '*'+list(Dx_map_unscored[Dx_map_unscored['SNOMED CT Code']==code]['Dx'])[0]
    return entry


def get_scored_class(code, labels):
    return [1 if label in code else 0 for label in labels]

def cv_split(headers_datasets, i_fold=0):
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
            filename = header_data[0].split(' ')[0].split('.')[0]
            data_labels = get_scored_class(codes, labels)

            Codes.append(codes)
            filenames.append(filename)
            
            dataset_data_labels[dataset].append(data_labels)
            dataset_idx[dataset].append(global_idx)
            global_idx += 1
        
        kf = MultilabelStratifiedKFold(5, random_state=0)
        kf_splits = kf.split(np.array(dataset_data_labels[dataset]), np.array(dataset_data_labels[dataset]))
        train_idx = None
        test_idx = None
        for _ in range(i_fold+1):
            train_idx, test_idx = next(kf_splits)

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
