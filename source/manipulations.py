
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
        