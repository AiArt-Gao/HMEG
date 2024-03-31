import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import torchvision.transforms as TF

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    from PIL import Image
    import os

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)


    class ImagePathDataset(torch.utils.data.Dataset):
        def __init__(self, files, transforms=None):
            self.files = files
            self.transforms = transforms

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            path = self.files[i]
            img = Image.open(path).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            return img



    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    tars_dir = [
        '../../datasets/results_test_src/test_npy_sub',
        '../../datasets/results_test_src/test_npy_nosub',
        '../../datasets/results_test_src/test_npy_grid_sub',
        '../../datasets/results_test_src/test_npy_grid_nosub',
        '../../datasets/results_test_src/test_nomask_nosub',
    ]

    # FILES = os.listdir('../../datasets/results/ours')

    for tar_dir in tars_dir:
        files = []
        for file in os.listdir(tar_dir):
            files.append(os.path.join(tar_dir, file))

        ds = ImagePathDataset(files, transforms=TF.ToTensor())

        print ("Calculating Inception Score...")
        print (inception_score(ds, cuda=True, batch_size=32, resize=True, splits=10))