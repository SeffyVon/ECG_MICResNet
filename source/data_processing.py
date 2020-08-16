from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == '__main__':
    
    Datas = []
    Header_datas = []
    Classes = []
    Codes = []
    
    dataset_idx = {}
    dataset_data_labels = {}
    dataset_train_idx = {}
    dataset_test_idx = {}
    
    global_idx = 0
    datasets = [1,2,3,4,5,6]
    for dataset in datasets:
        print('Dataset ', dataset)
        # Parse arguments.
        if len(sys.argv) != 3:
            raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

        input_directory = '../NewData/{}/'.format(dataset)
        output_directory = '../Output/'

        # Find files.
        input_files = []
        for f in os.listdir(input_directory):
            if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
                input_files.append(f)

        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        classes=get_classes(input_directory,input_files)

        num_files = len(input_files)
        datas = []
        header_datas = []
        dataset_data_labels[dataset] = []
        dataset_idx[dataset] = []
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