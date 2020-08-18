from saved_data_io import read_file, write_file
import os
from tqdm import tqdm

def pkl_split(headers_datasets):
    output_directory = 'saved'
    if not os.path.isdir(output_directory + '/cwt'):
            os.mkdir(output_directory+ '/cwt')

    for dataset in tqdm([1,2,3,4,5,6], desc='writing pkls', leave=False):
        headers = headers_datasets[dataset]
        data_imgs = read_file(output_directory +'/data_imgs_dataset{}.pkl'.format(dataset))
        for i, data_img in tqdm(enumerate(data_imgs), leave=False):
            filename = headers[i][0].split(' ')[0]
            write_file(output_directory +'/cwt/' + filename + '.pkl', data_img)

if __name__ == '__main__':
    output_directory = 'saved'
    filename = 'A0001'
    imgs = read_file(output_directory+ '/cwt/' + filename + '.pkl')
    print(len(imgs), type(imgs[0]))