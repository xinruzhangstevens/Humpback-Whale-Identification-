import torch
import numpy as np


def train_batch(batch):
    size = len(batch)

    labels = []
    image_batch = []

    for b in range(size):
        if batch[b][0]:
            image_batch.extend(batch[b][0])
            labels.extend(batch[b][1])

        image_batch = torch.stack(image_batch, 0)

    labels = np.array(labels)
    labels = torch.from_numpy(labels)

    return image_batch, labels


def valid_batch(batch):
    size = len(batch)
    labels = []
    image_batch = []
    image_names = []

    for b in range(size):
        if batch[b][0]:
            image_batch.extend(batch[b][0])
            labels.append(batch[b][1])
            image_names.append(batch[b][2])

    image_batch = torch.stack(image_batch, 0)
    labels = torch.from_numpy(np.array(labels))

    return image_batch, labels, image_names
