"""Module for training new CRNet models"""

import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch, LogStretch

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .util import maskMetric
from .dataset import dataset, PairedDatasetImagePath
from .unet import UNet, WrappedModel

__all__ = ["train"]


class VoidLRScheduler:
    def __init__(self):
        pass

    def _reset(self):
        pass

    def step(self, *args, **kwargs):
        pass
    
    
class train:
    def __init__(self, image, mask=None, ignore=None, sky=None, aug_sky=(0, 0),
                 name="model", hidden=32, model_kwargs=None, epoch=50, epoch_phase0=None, batch_size=16, lr=1e-3, 
                 auto_lr_decay=True, lr_decay_patience=4, lr_decay_factor=0.1, save_after=0, plot_every=10,
                 verbose=True, use_tqdm=False, use_tqdm_notebook=False, directory="./"):
        
        """ This is the class for training CRNet model.
        :param image: np.ndarray (N*W*W) of CR affected images or list of npy arrays of shape (2or3,W,W), where
        the 1st dim is CR affected image, 2nd dim is CR mask, (optional) 3rd dimension is ignore mask.
        :param mask: np.ndarray (N*W*W) training data: CR mask array.
        :param ignore: training data: Mask for taking loss. e.g., bad pixel, saturation, etc.
        :param sky: np.ndarray (N,) (optional) sky background. When param: iamge is list to npy files, provide sky level
        as sky.npy in each image subdirectory.
        :param aug_sky: [float, float]. If sky is provided, use random sky background in the range
          [aug_sky[0] * sky, aug_sky[1] * sky]. This serves as a regularizers to allow the trained model to adapt to a
          wider range of sky background or equivalently exposure time. Remedy the fact that exposure time in the
          training set is discrete and limited.
        :param name: model name. model saved to name.pth
        :param hidden: number of channels for the first convolution layer. default: 32
        :param gpu: True if use GPU for training
        :param epoch: Number of epochs to train. default: 50
        :param batch_size: training batch size. default: 16
        :param lr: learning rate. default: 0.005
        :param auto_lr_decay: reduce learning rate by "lr_decay_factor" after validation loss do not decrease for
          "lr_decay_patience" + 1 epochs.
        :param lr_decay_patience: reduce learning rate by lr_decay_factor after validation loss do not decrease for
          "lr_decay_patience" + 1 epochs.
        :param lr_decay_factor: multiplicative factor by which to reduce learning rate.
        :param save_after: epoch after which trainer automatically saves model state with lowest validation loss
        :param plot_every: for every "plot_every" epoch, plot mask prediction and ground truth for 1st image in
          validation set.
        :param verbose: print validation loss and detection rates for every epoch.
        :param use_tqdm: whether to show tqdm progress bar.
        :param use_tqdm_notebook: whether to use jupyter notebook version of tqdm. Overwrites tqdm_default.
        :param directory: directory relative to current path to save trained model.
        """
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = (self.device.type == "cuda")
        if not self.is_cuda:
            print("No GPU detected on this device! Training on CPU.")
            
        is_path = isinstance(image[0], (str, np.str_))
        data_train = (PairedDatasetImagePath(image, *aug_sky, part="train") if is_path
                      else dataset(image, mask, ignore, sky, part="train", aug_sky=aug_sky))
        data_val = (PairedDatasetImagePath(image, *aug_sky, part="val") if is_path
                    else dataset(image, mask, ignore, sky, part="val", aug_sky=aug_sky))

        cpu_cores = os.cpu_count() or 4
        num_workers = min(8, cpu_cores//2) if self.is_cuda else min(2, cpu_cores//2)

        dl_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=self.is_cuda, 
                     persistent_workers=(self.is_cuda and num_workers > 0),)

        self.TrainLoader = DataLoader(data_train, shuffle=True, **dl_kw)
        self.ValLoader = DataLoader(data_val, shuffle=False, **dl_kw)

        self.shape = data_train[0][0].shape[1]
        self.name, self.directory, self.verbose = name, directory, verbose
        self.every, self.save_after = plot_every, save_after
        self.n_epochs = epoch
        self.n_epochs_phase0 = int(self.n_epochs * 0.4 + 0.5) if epoch_phase0 is None else epoch_phase0
        
        mk = dict(n_channels=1, n_classes=1, hidden=hidden, act_norm="batch", att=False, deeper=False)

        if model_kwargs:
            mk.update(model_kwargs)   # user overrides anything above

        net = UNet(**mk)
        self.network = (nn.DataParallel(net) if self.device.type == "cuda" else WrappedModel(net)).to(self.device)

        self.lr = lr

        # changed optimizer to AdamW (minimal change)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-4)

        self._sched_cfg = dict(auto=auto_lr_decay, factor=lr_decay_factor, patience=lr_decay_patience)
        self.lr_scheduler = (ReduceLROnPlateau(
            self.optimizer, factor=lr_decay_factor, patience=lr_decay_patience,
            cooldown=2, verbose=True, threshold=0.005
        ) if auto_lr_decay else VoidLRScheduler())

        self.BCELoss = nn.BCELoss()
        self.validation_loss, self.best_loss = [], float("inf")
        self.epoch_mask = 0

        self.tqdm = tqdm_notebook if use_tqdm_notebook else tqdm
        self.disable_tqdm = not (use_tqdm_notebook or use_tqdm)
        self.writer = SummaryWriter(log_dir=directory)
        
        
    def _to4d(self, x):        
        return x.to(self.device, dtype=torch.float32, non_blocking=True).reshape(-1, 1, self.shape, self.shape)

    def set_input(self, img0, mask, ignore):
        """
        img0: input image tensor from DataLoader
        ignore: optional ignore mask (bad pixels, saturation, etc.)
        """
        
        img0 = (img0 - img0.mean()) / img0.std().clamp_min(1e-6)
        self.img0, self.mask, self.ignore = map(self._to4d, (img0, mask, ignore))

    def backward_network(self):
        return self.BCELoss(self.pdt_mask * (1 - self.ignore), self.mask * (1 - self.ignore))

    def optimize_network(self, dat):
        self.set_input(*dat)
        self.pdt_mask = self.network(self.img0)
        self.optimizer.zero_grad()
        loss = self.backward_network()
        loss.backward()
        self.optimizer.step()
        
    def validate_mask(self, epoch=None):
        """
        :return: validation loss. print TPR and FPR at threshold = 0.5.
        """        
        torch.random.manual_seed(0)
        np.random.seed(0)

        lmask = 0.0; count = 0
        metric = np.zeros(4)

        with torch.no_grad():
            for dat in self.ValLoader:
                n = dat[0].shape[0]
                count += n
                self.set_input(*dat)
                self.pdt_mask = self.network(self.img0)
                loss = self.backward_network()
                lmask += float(loss) * n
                pred = (self.pdt_mask.squeeze(1).detach().cpu().numpy() > 0.5)
                metric += maskMetric(pred, dat[1].numpy())

        lmask /= max(count, 1)
        TP, TN, FP, FN = metric
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        if self.verbose:
            print("[val_loss=%.6f, TPR=%.3f, FPR=%.3f] @threshold = 0.5" % (lmask, TPR, FPR))

        if epoch:  
            self.writer.add_scalar("TPR", TPR, epoch)
            self.writer.add_scalar("FPR", FPR, epoch)
            self.writer.add_scalar("validate_loss", lmask, epoch)

        return lmask

    def _run(self, epochs, eval_mode=False, reset_sched=False):
        self.network.eval() if eval_mode else self.network.train()
        if reset_sched:
            self.lr_scheduler._reset()

        for epoch in self.tqdm(range(epochs), disable=self.disable_tqdm):
            for dat in self.TrainLoader:
                self.optimize_network(dat)

            self.epoch_mask += 1
            if self.epoch_mask % self.every == 0:
                self.plot_example()

            if self.verbose:
                print("----------- epoch = %d -----------" % self.epoch_mask)

            val = self.validate_mask(epoch)
            self.validation_loss.append(val)
            
            if val <= self.best_loss:
                self.best_loss = val
                if self.epoch_mask > self.save_after:
                    fn = self.save()
                    if self.verbose:
                        print("Saved to {}.pth".format(fn))

            self.lr_scheduler.step(val)
            if self.verbose:
                print("")

    def train(self):
        if self.verbose:
            print(f"Begin first {self.n_epochs_phase0} epochs of training")
            print("Use batch statistics for batch normalization; keep running statistics to be used in phase1\n")
        self.train_phase0(self.n_epochs_phase0)

        fn = self.save()
        self.load(fn)

        # AdamW here too 
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self.lr / 2.5, weight_decay=1e-4)

        if self._sched_cfg["auto"]:
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                factor=self._sched_cfg["factor"],
                patience=self._sched_cfg["patience"],
                cooldown=2,
                verbose=True,
                threshold=0.005,
            )
        else:
            self.lr_scheduler = VoidLRScheduler()

        if self.verbose:
            print(f"Continue onto next {self.n_epochs - self.n_epochs_phase0} epochs of training")
            print("Batch normalization running statistics frozen and used\n")
        self.train_phase1(self.n_epochs - self.n_epochs_phase0)

    def train_phase0(self, epochs):
        self._run(epochs, eval_mode=False, reset_sched=False)

    def train_phase1(self, epochs):
        self._run(epochs, eval_mode=True, reset_sched=True)

    def plot_example(self):
        interval = ZScaleInterval()
        true_img = self.img0[0, 0].detach().cpu().numpy()
        vmin, vmax = interval.get_limits(true_img)
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())

        plt.figure(figsize=(15, 5))
        plt.subplot(131); plt.imshow(true_img, cmap="gray", norm=norm, origin='lower'); plt.title(f"epoch={self.epoch_mask}")
        plt.subplot(132); plt.imshow(self.pdt_mask[0, 0].detach().cpu().numpy() > 0.5, cmap="gray", origin='lower'); plt.title("prediction > 0.5")
        plt.subplot(133); plt.imshow(self.mask[0, 0].detach().cpu().numpy(), cmap="gray", origin='lower'); plt.title("ground truth")

        print("Save trainplot")
        plt.savefig(self.directory + self.name + f"epoch{self.epoch_mask}trainplot.png")
        plt.close()

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(self.epoch_mask), self.validation_loss)
        plt.grid()        
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Validation loss")
        plt.savefig(self.directory + self.name + "trainplot.png")
        plt.close()

    # save model 
    def save(self):
        torch.save(self.network.state_dict(), self.directory + self.name + ".pth")
        return self.name
    
    def load(self, filename):
        self.network.load_state_dict(
            torch.load(self.directory + filename + ".pth", map_location=self.device)
        )
        