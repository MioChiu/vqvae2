import os
import random
import torch.utils.data as data
from PIL import Image


class MvtecDataset(data.Dataset):
    def __init__(self, root, train=True, classes=None, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.classes = classes
        self.transform = transform
        self.target_transform = target_transform     
        self.category_dict = {
            'bottle': 0,
            'cable': 1,
            'capsule': 2,
            'carpet': 3,
            'grid': 4,
            'hazelnut': 5,
            'leather': 6,
            'metal_nut': 7,
            'pill': 8,
            'screw': 9,
            'tile': 10,
            'toothbrush': 11,
            'transistor': 12,
            'wood': 13,
            'zipper': 14
        }

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        else:
            self._check_txt()

        self.data_list = []
        if self.train:
            if self.classes is None:
                self.classes = list(self.category_dict.keys())
            assert set(self.classes) <= set(list(self.category_dict.keys()))

            for c in self.classes:
                cls_path = os.path.join(self.root, 'txt', c + '_train.txt')
                with open(cls_path, 'r') as f:
                    lines = f.readlines()
                self.data_list.extend(lines)
        else:
            if self.classes is None:
                self.classes = self.category_dict.keys()
            assert self.classes in self.category_dict.keys()
            for c in self.classes:
                cls_path = os.path.join(self.root, 'txt', c + '_test.txt')
                with open(cls_path, 'r') as f:
                    lines = f.readlines()
                self.data_list.extend(lines)
        random.shuffle(self.data_list)

    def __getitem__(self, index):
        img_path, label = self.data_list[index].strip().split(' ')
        img = Image.open(img_path)
        img = img.convert('RGB')
        target = self.category_dict[label]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_list)

    def _check_exists(self):
        return os.path.exists(self.root)

    def _check_txt(self):
        if not os.path.exists(os.path.join(self.root, 'txt')):
            os.mkdir(os.path.join(self.root, 'txt'))
        for c in self.category_dict.keys():
            for split in ['train', 'test']:
                cls_split_path = os.path.join(self.root, 'txt', c + '_' + split + '.txt')
                if not os.path.exists(cls_split_path):
                    print('{} is not found.'.format(cls_split_path))
                    with open(cls_split_path, 'w') as f:
                        for home, dirs, files in os.walk(os.path.join(self.root, c, split)):
                            for filename in files:
                                if 'png' in filename:
                                    img_path = os.path.join(home, filename)
                                    f.write(img_path + ' ' + c + '\n')

        # for split in ['train', 'test']:
        #     all_spilt_txt = os.path.join(self.root, split + '.txt')
        #     if not os.path.exists(all_spilt_txt):
        #         print('{} is not found.'.format(all_spilt_txt))
        #         train_list = []
        #         for c in self.category_dict.keys():
        #             cls_split_path = os.path.join(self.root, c + '_' + split + '.txt')
        #             with open(cls_split_path, 'r') as f:
        #                 lines = f.readlines()
        #                 train_list.extend(lines)
        #         with open(all_spilt_txt, 'w') as f:
        #             f.writelines(train_list)


if __name__ == '__main__':
    dataset = MvtecDataset('/mnt/qiuzheng/data/mvtec')
