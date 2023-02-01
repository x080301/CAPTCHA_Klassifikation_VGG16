import pandas as pd
import os
import numpy as np

import torch
from skimage.io import imread
import torchvision.transforms as transforms


def writecsv(data_dir, train_num_per_class=2000):
    column = ['filename;Bicycle;Bridge;Bus;Car;Chimney;Crosswalk;Hydrant;Motorcycle;Other;Palm;Stair;Traffic Light']

    labels = ['Bicycle', 'Bridge', 'Bus', 'Car', 'Chimney', 'Crosswalk', 'Hydrant', 'Motorcycle', 'Other', 'Palm',
              'Stair', 'Traffic Light']

    one_hot = np.eye(12).astype(int).astype(str)

    train_list = []
    valid_list = []

    directory_list = os.listdir(data_dir)
    for directory in directory_list:

        label = labels.index(directory)

        label_onhot = ';'.join(list(one_hot[label, :]))

        directory_dir = data_dir + '/' + directory

        i = 0
        train_list_block = []
        for file_name in os.listdir(directory_dir):

            data_row = directory_dir + '/' + file_name + ';' + label_onhot
            if i % 4 == 0:

                valid_list.append(data_row)
            else:
                train_list_block.append(data_row)

            i = i + 1

        j = 0
        while j <= train_num_per_class:

            for data_row in train_list_block:
                j += 1
                if j > train_num_per_class:
                    break
                else:
                    train_list.append(data_row)

    traincsv = pd.DataFrame(columns=column, data=train_list)
    traincsv.to_csv('data_train.csv', index=False)

    validcsv = pd.DataFrame(columns=column, data=valid_list)
    validcsv.to_csv('data_valid.csv', index=False)


def get_mean_std(data_dir):
    '''
    Compute mean and variance for training data
    :param train_data:
    :return: (mean, std)
    '''

    transf = transforms.ToTensor()

    mean = torch.zeros(3)
    std = torch.zeros(3)

    length = 0
    for class_name in os.listdir(data_dir + '/train_val'):
        directory_dir = data_dir + '/train_val/' + class_name
        for file_name in os.listdir(directory_dir):
            image = imread(directory_dir + '/' + file_name)
            image = transf(image)

            length += 1
            for d in range(3):
                mean[d] += image[d, :, :].mean()
                std[d] += image[d, :, :].std()

    return mean.div_(length), std.div_(length)


def writecsv_test(data_dir):
    column = ['filename;filename2']

    data_list = []
    for file_name in os.listdir(data_dir):
        data_row = data_dir + '/' + file_name + ';' + file_name
        data_list.append(data_row)
    test_csv = pd.DataFrame(columns=column, data=data_list)
    test_csv.to_csv('data_test.csv', index=False)


if __name__ == "__main__":
    writecsv_test('dataset/test')
    # writecsv('dataset/train_val', train_num_per_class=2000)

    # print(get_mean_std('dataset'))  # reslut: (tensor([0.4795, 0.4722, 0.4359]), tensor([0.1675, 0.1676, 0.1834]))
    pass
