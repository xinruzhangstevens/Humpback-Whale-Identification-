from torch.utils.data import Dataset
import pandas as pd
import random
import cv2
import numpy as np


def do_length_decode(rle, H=192, W=384, fill_value=255):
    mask = np.zeros((H, W), np.uint8)
    if type(rle).__name__ == 'float':
        return mask
    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start: end] = fill_value
    mask = mask.reshape(W, H).T   # H, W need to swap as transposing.
    return mask


class Trainset(Dataset):
    def __init__(self, names, labels=None, mode='train', transform_train=None,  min_num_classes=0):
        super(Trainset, self).__init__()
        self.pairs = 2
        self.names = names
        self.labels = labels
        self.mode = mode
        self.transform_train = transform_train
        self.labels_dict = self.load_label()
        self.bbox_dict = self.load_bbox()
        self.rle_masks = self.load_mask()
        self.id_labels = {name: label for name, label in zip(self.names, self.labels)}

        self.dict_train = self.label_nameList()
        self.labels = [label for label in self.dict_train.keys() if len(self.dict_train[label]) >= min_num_classes]

    def load_mask(self):
        masks = pd.read_csv('./mask.csv')
        if masks:
            print('masks loaded...')
        else:
            print('masks loaded wrong...')

        loc_notnull = masks[masks['rle_mask'].isnull().values == False].index.tolist()
        masks = masks[loc_notnull]
        masks.index = masks['id']
        masks.drop(['id'])
        masks = masks.to_dict('index')

        return masks

    def load_bbox(self):
        # Image,x0,y0,x1,y1
        bbox = pd.read_csv('./bboxs.csv')
        if bbox:
            print('bounding box loaded')
        else:
            print('bounding box loaded wrong')

        Images = bbox['Image'].tolist()
        x0_list = bbox['x0'].tolist()
        y0_list = bbox['y0'].tolist()
        x1_list = bbox['x1'].tolist()
        y1_list = bbox['y1'].tolist()
        bbox_dict = {}

        assert len(Images) == len(x0_list)
        assert len(Images) == len(x1_list)
        assert len(Images) == len(y0_list)
        assert len(Images) == len(y1_list)

        for i in range(len(Images)):
            bbox_dict[Images[i]] = [x0_list[i], y0_list[i], x1_list[i], y1_list[i]]

        return bbox_dict

    def load_label(self):
        labels = pd.read_csv('./label.csv')
        if labels:
            print('labels loaded')
        else:
            print('bounding box loaded wrong')

        labelNames = labels['name'].tolist()
        label_dict = {}
        class_id = 0

        for name in labelNames:
            if name in label_dict.keys():
                continue
            elif name == 'new_whale':
                label_dict[name] = 5004 * 2
                continue

            label_dict[name] = class_id
            class_id += 1

        return label_dict

    def label_nameList(self):
        label_names = {}
        for name, label in zip(self.names, self.labels):
            if label not in label_names.keys():
                label_names[label] = [name]
            else:
                label_names[label].append(name)

        assert len(self.names) == len(self.labels), 'length of names and labels is not equal...'

        for i in range(len(self.names)):
            if self.labels[i] not in label_names.keys():
                label_names[self.labels[i]] = [self.names[i]]
            else:
                label_names[self.labels[i]].append(self.names[i])

        return label_names

    def __len__(self):
        num_class = len(self.labels)
        return num_class

    def get_image(self, name, transform, label, mode='train'):
        image = cv2.imread('./{}/{}'.format(mode, name))
        # for Pseudo label
        if image is None:
            image = cv2.imread('./test/{}'.format(name))
        try:
            mask = do_length_decode(self.rle_masks[name.split('.')[0]]['rle_mask'])
            mask = cv2.resize(mask, image.shape[:2][::-1])
        except:
            mask = cv2.imread('./masks/' + name, cv2.IMREAD_GRAYSCALE)  # Read images in grayscale mode。
        x0, y0, x1, y1 = self.bbox_dict[name]
        if mask is None:
            mask = np.zeros_like(image[:, :, 0])
        image = image[int(y0):int(y1), int(x0):int(x1)]
        mask = mask[int(y0):int(y1), int(x0):int(x1)]
        image, add_ = transform(image, mask, label)
        return image, add_

    def __getitem__(self, index):
        label = self.labels[index]
        namelist = self.dict_train[label]
        if len(namelist) == 1:
            anchor_name = namelist[0]
            positive_name = namelist[0]
        else:
            anchor_name, positive_name = random.sample(namelist, 2)

        negative_label = random.choice(list(set(self.labels) ^ set([label, 'new_whale'])))  # ^两个集合的对称差（去重后放在一起）
        negative_name = random.choice(self.dict_train[negative_label])
        negative_label2 = 'new_whale'
        negative_name2 = random.choice(self.dict_train[negative_label2])

        anchor_image, anchor_add = self.get_image(anchor_name, self.transform_train, label)
        positive_image, positive_add = self.get_image(positive_name, self.transform_train, label)
        negative_image,  negative_add = self.get_image(negative_name, self.transform_train, negative_label)
        negative_image2, negative_add2 = self.get_image(negative_name2, self.transform_train, negative_label2)

        assert anchor_name != negative_name
        return [anchor_image, positive_image, negative_image, negative_image2], \
               [self.labels_dict[label] + anchor_add, self.labels_dict[label] + positive_add,
                self.labels_dict[negative_label] + negative_add, self.labels_dict[negative_label2] + negative_add2]


class Testset(Dataset):
    def __init__(self, names, labels=None, mode='test', transform=None):
        super(Testset, self).__init__()
        self.names = names
        self.labels = labels
        self.mode = mode
        self.bbox_dict = self.load_bbox()
        self.labels_dict = self.load_label()
        self.rle_masks = self.load_mask()
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def get_image(self, name, transform, mode='train'):
        image = cv2.imread('./{}/{}'.format(mode, name))
        try:
            mask = do_length_decode(self.rle_masks[name.split('.')[0]]['rle_mask'])
            mask = cv2.resize(mask, image.shape[:2][::-1])
        except:
            mask = cv2.imread('./masks/' + name, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros_like(image[:, :, 0])
        x0, y0, x1, y1 = self.bbox_dict[name]
        image = image[int(y0):int(y1), int(x0):int(x1)]
        mask = mask[int(y0):int(y1), int(x0):int(x1)]
        image = transform(image, mask)
        return image

    def load_mask(self):
        masks = pd.read_csv('./mask.csv')
        if masks:
            print('masks loaded...')
        else:
            print('masks loaded wrong...')

        loc_notnull = masks[masks['rle_mask'].isnull().values == False].index.tolist()
        masks = masks[loc_notnull]
        masks.index = masks['id']
        masks.drop(['id'])
        masks = masks.to_dict('index')

        return masks

    def load_bbox(self):
        # Image,x0,y0,x1,y1
        bbox = pd.read_csv('./bboxs.csv')
        if bbox:
            print('bounding box loaded')
        else:
            print('bounding box loaded wrong')

        Images = bbox['Image'].tolist()
        x0_list = bbox['x0'].tolist()
        y0_list = bbox['y0'].tolist()
        x1_list = bbox['x1'].tolist()
        y1_list = bbox['y1'].tolist()
        bbox_dict = {}

        assert len(Images) == len(x0_list)
        assert len(Images) == len(x1_list)
        assert len(Images) == len(y0_list)
        assert len(Images) == len(y1_list)

        for i in range(len(Images)):
            bbox_dict[Images[i]] = [x0_list[i], y0_list[i], x1_list[i], y1_list[i]]

        return bbox_dict

    def load_label(self):
        labels = pd.read_csv('./label.csv')
        if labels:
            print('labels loaded')
        else:
            print('bounding box loaded wrong')

        labelNames = labels['name'].tolist()
        label_dict = {}
        class_id = 0

        for name in labelNames:
            if name in label_dict.keys():
                continue
            elif name == 'new_whale':
                label_dict[name] = 5004 * 2
                continue

            label_dict[name] = class_id
            class_id += 1

        return label_dict

    def __getitem__(self, index):
        if self.mode in ['test']:
            name = self.names[index]
            image = self.get_image(name, self.transform, mode='test')
            return image, name
        elif self.mode in ['valid', 'train']:
            name = self.names[index]
            label = self.labels_dict[self.labels[index]]
            image = self.get_image(name, self.transform)
            return image, label, name