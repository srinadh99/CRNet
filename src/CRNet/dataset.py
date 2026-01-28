import numpy as np
from torch.utils.data import Dataset
import os

__all__ = ['dataset', 'PairedDatasetImagePath']


class dataset(Dataset):
    def __init__(self, image, mask, ignore=None, sky=None, aug_sky=[0, 0], part=None, f_val=0.1, seed=1):

        """ custom pytorch dataset class to load training data
        :param image: image with CR. Could be (N, W, H) array or list of path to single (W, H) images.
        :param mask: CR mask. Could be (N, W, H) array or list of path to single (W, H) images.
        :param ignore: (optional) loss mask, e.g., bad pixel, saturation, etc. Could be (N, W, H) array or list
        of path to single (W, H) images
        :param sky: (np.ndarray) [N,] sky background level
        :param aug_sky: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. 
        :param part: either 'train' or 'val'. split by 0.9, 0.1
        :param f_val: percentage of dataset reserved as validation set.
        :param seed: fix numpy random seed to seed, for reproducibility.
        """

        np.random.seed(seed)
        len = image.shape[0]
        assert f_val < 1 and f_val > 0
        f_train = 1 - f_val
        if sky is None:
            sky = np.zeros_like(image)
        if ignore is None:
            ignore = np.zeros_like(image)

        if part == 'train':
            s = np.s_[:max(1, int(len * f_train))]
        elif part == 'val':
            s = np.s_[min(len - 1, int(len * f_train)):]
        else:
            s = np.s_[0:]

        self.image = image[s]
        self.mask = mask[s]
        self.ignore = ignore[s]
        self.sky = sky[s]

        self.aug_sky = aug_sky

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, i):
        a = (self.aug_sky[0] + np.random.rand() * (self.aug_sky[1] - self.aug_sky[0])) * self.sky[i]
        ignore = self.ignore[i] if type(self.ignore[i]) != str else np.load(self.ignore[i])
        if type(self.image[i]) == str:
            return np.load(self.image[i]) + a, np.load(self.mask[i]), ignore
        else:
            return self.image[i] + a, self.mask[i], ignore


class PairedDatasetImagePath(Dataset):
    def __init__(self, paths, skyaug_min=0, skyaug_max=0, part=None, f_val=0.1, seed=1):

        """ custom pytorch dataset class to load training data
        :param paths: (list) list of file paths to (3, W, H) images: image, cr, ignore.
        :param skyaug_min: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param skyaug_min: float. subtract maximum amount of abs(skyaug_min) * sky_level as data augmentation
        :param skyaug_max: float. add maximum amount of skyaug_max * sky_level as data augmentation
        :param part: either 'train' or 'val'.
        :param f_val: percentage of dataset reserved as validation set.
        :param seed: fix numpy random seed to seed, for reproducibility.
        """

        assert 0 < f_val < 1
        np.random.seed(seed)
        n_total = len(paths)
        n_train = int(n_total * (1 - f_val)) 

        if part == 'train':
            s = np.s_[:max(1, n_train)]
        elif part == 'val':
            s = np.s_[min(n_total - 1, n_train):]
        else:
            s = np.s_[0:]

        self.paths = paths[s]
        self.skyaug_min = skyaug_min
        self.skyaug_max = skyaug_max

    def __len__(self):
        return len(self.paths)

    def get_skyaug(self, i):

        """
        Return the amount of background flux to be added to image
        The original sky background should be saved in sky.npy in each sub-directory
        Otherwise always return 0
        :param i: index of file
        :return: amount of flux to add to image
        """

        path = os.path.split(self.paths[i])[0]
        sky_path = os.path.join(path, 'sky.npy')
        if os.path.isfile(sky_path):
            f_img = self.paths[i].split('/')[-1]
            sky_idx = int(f_img.split('_')[0])
            sky = np.load(sky_path)[sky_idx-1]
            return sky * np.random.uniform(self.skyaug_min, self.skyaug_max)
        else:
            return 0

    def __getitem__(self, i):
        data = np.load(self.paths[i])
        image = data[0]
        mask = data[1]
        if data.shape[0] == 3:
            ignore = data[2]
        else:
            ignore = np.zeros_like(data[0])
        skyaug = self.get_skyaug(i)
        return image + skyaug, mask, ignore
