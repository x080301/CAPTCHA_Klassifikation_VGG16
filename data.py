from torch.utils.data import Dataset, DataLoader
import torch
from skimage.io import imread
import torchvision.transforms as transforms
import pandas as pd


class CAPTCHADataset(Dataset):
    def __init__(self, data, mode):
        super(CAPTCHADataset, self).__init__()

        self.data = data
        self.mode = mode

        _train_mean = [0.4795, 0.4722, 0.4359]
        _train_std = [0.1675, 0.1676, 0.1834]
        _size = 120

        if mode == 'train':
            self.transform = transforms.Compose([

                transforms.ToPILImage(),

                transforms.Resize((int(_size * 1.25), int(_size * 1.25))),  # 防止旋转后边界出现黑框部分
                transforms.RandomRotation(15),
                transforms.CenterCrop(_size),  # 防止旋转后边界出现黑框部分

                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),

                transforms.ToTensor(),
                transforms.Normalize(mean=_train_mean,
                                     std=_train_std),
                transforms.RandomErasing(),
                transforms.GaussianBlur(5)
            ])
        else:  # they can be different.
            self.transform = transforms.Compose([

                transforms.ToPILImage(),
                transforms.Resize(_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_train_mean,
                                     std=_train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image_dir = self.data.iloc[index, 0]
        # image_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], image_dir)
        image = imread(image_dir)
        image = image[:, :, 0:3]
        image = self.transform(image)

        #
        if self.mode == 'test':
            label = self.data.iloc[index, 1]
        else:
            label = self.data.iloc[index, 1:]
            label = torch.tensor(label, dtype=torch.long)

        return image, label


if __name__ == '__main__':
    train_dataset = pd.read_csv('test.csv', sep=';')
    # print(train_dataset.iloc[1,0])

    db = CAPTCHADataset(train_dataset, 'train')
    loader = DataLoader(db, batch_size=32, shuffle=True)
    # print(loader)

    for x, y in loader:
        print(y)
        print(x)
