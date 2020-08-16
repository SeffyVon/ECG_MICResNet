
import numpy as np, os, sys
from scipy.io import loadmat
from tqdm.notebook import tqdm

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

"""
"""

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def get_Datas(datasets=[1,2,3,4,5,6]):
    """
    Read data from input directory
    Return:
    -- first pass
    Datas: list
    Header_datas: list
    Classes: list
    dataset_idx: dict
    --- stratified ----
    dataset_train_idx: dict
    dataset_test_idx: dict
    """
    
    Datas = []
    Header_datas = []
    Classes = []
    Codes = []
    
    dataset_idx = {}
    dataset_data_labels = {} # encoding
    dataset_train_idx = {}
    dataset_test_idx = {}
    
    global_idx = 0
    for dataset in datasets:
        print('Dataset ', dataset)

        input_directory = '../NewData/{}/'.format(dataset)
        # Find files.
        input_files = []
        for f in os.listdir(input_directory):
            if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
                input_files.append(f)

        classes=get_classes(input_directory,input_files)

        num_files = len(input_files)
        datas = []
        header_datas = []
        dataset_idx[dataset] = []
        dataset_data_labels[dataset] = []
        for i, f in tqdm(enumerate(input_files)):
            #print('    {}/{}...'.format(i+1, num_files), f)
            tmp_input_file = os.path.join(input_directory,f)
            data,header_data = load_challenge_data(tmp_input_file)
            
            codes = get_classes_from_header(header_data)
            data_labels = get_scored_class(codes, labels)
            Codes.append(codes)
            
            datas.append(data[:,1000:7000])
            header_datas.append(header_data)
            dataset_data_labels[dataset].append(data_labels)
            dataset_idx[dataset].append(global_idx)
            global_idx += 1

        Datas += datas
        Header_datas += header_datas
        Classes += classes
        
        kf = MultilabelStratifiedKFold(5, random_state=0)
        train_idx, test_idx = next(kf.split(datas, np.array(dataset_data_labels[dataset])))

        dataset_train_idx[dataset] = train_idx +  dataset_idx[dataset][0]
        dataset_test_idx[dataset] = test_idx + dataset_idx[dataset][0]
        
        print('Done.')
    return Datas, Codes, dataset_data_labels, dataset_train_idx, dataset_test_idx

from signal_processing import myfilter, main_QRST
def segment_QRS(Datas):
    """
    Segment to QRS
    Input: Datas, array
    Output: Q_locs, array
    """
    Q_locs = []
    for idx in tqdm(range(0, len(Datas))): 
        codes = get_classes_from_header(Header_datas[idx])
        names = ', '.join([get_name(int(code), Dx_map, Dx_map_unscored) for code in codes])

        filtered_Data = myfilter(Datas[idx], 500, vis=False)

        # get the lead to apply Pan Tomkins
        Q_loc = main_QRST(filtered_Data, idx, '', '', names, fig2=False)

        # store
        Q_locs.append(Q_loc)

    return Q_locs

from global_vars import labels, normal_class, equivalent_mapping

normal_idx = np.argwhere(labels==int(normal_class))
def get_scored_class(code, labels):
    return [1 if label in code else 0 for label in labels]