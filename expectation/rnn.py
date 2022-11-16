from abc import ABC, abstractmethod
import logging
import os

from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from tqdm import tqdm
import glob
import partitura
import warnings

warnings.filterwarnings("ignore")

LOGGER = logging.getLogger(__name__)


def load_data(min_seq_length: int = 10) -> List[np.ndarray]:
    # load data
    files = glob.glob(os.path.join("data", "*.mid"))
    files.sort()
    sequences = []
    for fn in files:
        seq = partitura.load_performance_midi(fn)[0]
        if len(seq.notes) > min_seq_length:
            sequences.append(seq.note_array())
    return sequences


class NNTrainer(ABC):
    """
    Class for training neural networks
    """

    def __init__(
        self,
        model,
        train_loss,
        optimizer,
        train_dataloader,
        valid_loss=None,
        valid_dataloader=None,
        best_comparison="smaller",
        n_gpu=1,
        epochs=100,
        save_freq=10,
        early_stopping=100,
        out_dir=".",
        resume_from_saved_model=None,
    ):

        self.n_gpu = n_gpu
        self.device, self.dtype = self.prepare_device()
        self.model = model.to(self.device)
        self.out_dir = out_dir

        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.start_epoch = 0
        self.epochs = epochs
        self.save_freq = save_freq
        self.early_stopping = early_stopping
        self.best_loss = np.inf
        self.best_comparison = best_comparison

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.loss_progress = {"train_loss": [], "val_loss": []}

        if resume_from_saved_model is not None:
            self.resume_checkpoint(resume_from_saved_model)

        self.batch_first = True

        if hasattr(self.model, "batch_first"):
            self.batch_first = self.model.batch_first

    @abstractmethod
    def train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def valid_step(self, *args, **kwargs):
        pass

    def train(self):

        train_loss_name = getattr(self.train_loss, "name", "Train Loss")
        train_fn = os.path.join(self.out_dir, "train_loss.txt")
        # Initialize TrainProgressMonitors
        train_losses = TrainProgressMonitor(train_loss_name, fn=train_fn)
        valid_loss_name = None
        valid_losses = None
        if self.valid_dataloader is not None:
            valid_fn = os.path.join(self.out_dir, "valid_loss.txt")
            if isinstance(self.valid_loss, (list, tuple)):
                valid_loss_name = [
                    getattr(crit, "name", "Valid Loss {0}".format(i))
                    for i, crit in enumerate(self.valid_loss)
                ]
            else:
                valid_loss_name = [
                    getattr(
                        self.valid_loss,
                        "name",
                        "Valid Loss",
                    )
                ]
            valid_losses = TrainProgressMonitor(
                valid_loss_name, fn=valid_fn, show_epoch=False
            )

        validations_wo_improvement = 0

        # save before training
        self.save_checkpoint(-1, False, True)
        try:
            for epoch in range(self.start_epoch, self.epochs):
                tl = self.train_step(epoch)

                train_losses.update(epoch, tl)

                do_checkpoint = np.mod(epoch + 1, self.save_freq) == 0

                if do_checkpoint:
                    if self.valid_dataloader is not None:
                        vl = self.valid_step(epoch)
                        valid_losses.update(epoch, vl)
                        LOGGER.info(
                            train_losses.last_loss + "\t" + valid_losses.last_loss
                        )
                    else:
                        vl = [tl]
                        LOGGER.info(train_losses.last_loss)
                    if self.best_comparison == "smaller":
                        is_best = vl[0] < self.best_loss
                        self.best_loss = min(vl[0], self.best_loss)
                    elif self.best_comparison == "larger":
                        is_best = vl[0] > self.best_loss
                        self.best_loss = max(vl[0], self.best_loss)

                    self.save_checkpoint(epoch, validate=do_checkpoint, is_best=is_best)

                    if is_best:
                        validations_wo_improvement = 0
                    else:
                        validations_wo_improvement += self.save_freq

                    if validations_wo_improvement > self.early_stopping:
                        break
        except KeyboardInterrupt:
            LOGGER.info("Training interrupted")
            pass

        # Load best model
        best_backup = torch.load(os.path.join(self.out_dir, "best_model.pth"))
        LOGGER.info(
            "Loading best model (epoch {0}: {1:.4f})".format(
                best_backup["epoch"], best_backup["best_loss"]
            )
        )
        self.model.load_state_dict(best_backup["state_dict"])

    def save_checkpoint(self, epoch, validate, is_best):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }
        if validate:
            filename = os.path.join(
                self.out_dir, "checkpoint-epoch-{}.pth".format(epoch)
            )
            torch.save(state, filename)
            LOGGER.info("Saving checkpoint: {0} ...".format(filename))
        if is_best:
            filename = os.path.join(self.out_dir, "best_model.pth")
            torch.save(state, filename)
            LOGGER.info("Saving current best: {0} ...".format(filename))

    def prepare_device(self):
        n_gpu = torch.cuda.device_count()
        if self.n_gpu > 0 and n_gpu == 0:
            LOGGER.warning("No GPU available! Training will be performed on CPU.")
            self.n_gpu = 0
        if self.n_gpu > n_gpu:
            LOGGER.warning(
                "Only {0} GPUs availabe. " "(`n_gpu` is {1})".format(n_gpu, self.n_gpu)
            )
            self.n_gpu = n_gpu

        device = torch.device("cuda:0" if self.n_gpu > 0 else "cpu")
        if device.type == "cuda":
            dtype = "torch.cuda.FloatTensor"
        else:
            dtype = "torch.FloatTensor"

        return device, dtype

    def resume_checkpoint(self, checkpoint_fn):
        LOGGER.info("Loading checkpoint: {0} ...".format(checkpoint_fn))
        checkpoint = torch.load(checkpoint_fn)
        self.start_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        LOGGER.info(
            "Checkpoint loaded. "
            "Resume training from epoch {0}".format(self.start_epoch)
        )


class TrainProgressMonitor(object):
    """
    Monitor the progress of training a model.
    """

    def __init__(
        self,
        name="",
        fn="train_progres.txt",
        show_fmt="{0} {1:.3f}",
        write_fmt="{0:.8f}",
        show_epoch=True,
    ):
        self.name = name
        self.losses = []
        self.epochs = []
        self.fn = fn
        self.show_fmt = show_fmt
        self.write_fmt = write_fmt
        self.show_epoch = show_epoch

        try:
            os.unlink(self.fn)
        except OSError:
            pass

        with open(self.fn, "w") as f:
            if isinstance(self.name, (list, tuple)):
                header = "# Epoch" + "\t" + "\t".join(self.name) + "\n"
            else:
                header = "# Epoch" + "\t" + self.name + "\n"

            f.write(header)

    def update(self, epoch, loss):
        """
        Append new loss(es) and update the log file
        """
        self.losses.append(loss)

        self.epochs.append(epoch)

        self.update_log()

    @property
    def last_loss(self):

        fmt = self.show_fmt

        if isinstance(self.losses[-1], (list, tuple, np.ndarray)):
            out_str = [fmt.format(n, l) for n, l in zip(self.name, self.losses[-1])]
        else:
            out_str = [fmt.format(self.name, self.losses[-1])]

        if self.show_epoch:
            return "Epoch:{0}\t{1}".format(self.epochs[-1], "\t".join(out_str))
        else:
            return "\t".join(out_str)

    def update_log(self):

        if isinstance(self.losses[-1], (list, tuple, np.ndarray)):
            out_str = "\t".join([self.write_fmt.format(l) for l in self.losses[-1]])
        else:
            out_str = self.write_fmt.format(float(self.losses[-1]))

        with open(self.fn, "a") as f:
            f.write("{0}\t{1}\n".format(self.epochs[-1], out_str))


class SupervisedTrainer(NNTrainer):
    def __init__(
        self,
        model,
        train_loss,
        optimizer,
        train_dataloader,
        valid_loss=None,
        valid_dataloader=None,
        best_comparison="smaller",
        lr_scheduler=None,
        n_gpu=1,
        epochs=10,
        save_freq=10,
        early_stopping=100,
        out_dir=".",
        resume_from_saved_model=None,
    ):
        super().__init__(
            model=model,
            train_loss=train_loss,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            valid_loss=valid_loss,
            valid_dataloader=valid_dataloader,
            best_comparison=best_comparison,
            n_gpu=n_gpu,
            epochs=epochs,
            save_freq=save_freq,
            early_stopping=early_stopping,
            out_dir=out_dir,
            resume_from_saved_model=resume_from_saved_model,
        )
        self.lr_scheduler = lr_scheduler

    def train_step(self, epoch, *args, **kwargs):

        self.model.train()
        losses = []
        bar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
        )
        for b_idx, (input, target) in bar:

            if self.device is not None:
                input = input.to(self.device).type(self.dtype)
                target = target.to(self.device).type(self.dtype)

            if not self.batch_first:
                input = input.transpose(0, 1)
                target = target.transpose(0, 1)

            # import pdb
            # pdb.set_trace()

            output = self.model(input)
            loss = self.train_loss(output, target)
            losses.append(loss.item())
            bar.set_description(
                "epoch: {}/{} loss: {:.2f}".format(
                    epoch + 1,
                    self.epochs,
                    losses[-1],
                )
            )

            if np.isnan(losses[-1]):
                import pdb

                pdb.set_trace()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return np.mean(losses)

    def valid_step(self, *args, **kwargs):

        self.model.eval()
        losses = []
        with torch.no_grad():
            for input, target in self.valid_dataloader:

                if self.device is not None:
                    target = target.to(self.device).type(self.dtype)
                    input = input.to(self.device).type(self.dtype)

                if not self.batch_first:
                    input = input.transpose(0, 1)
                    target = target.transpose(0, 1)

                output = self.model(input)

                if isinstance(self.valid_loss, (list, tuple)):
                    loss = [c(output, target) for c in self.valid_loss]
                else:
                    loss = [self.valid_loss(output, target)]
                losses.append([l.item() for l in loss])

        return np.mean(losses, axis=0)


class MSELoss(nn.Module):
    name = "MSE"

    def __call__(self, predictions, targets):
        return functional.mse_loss(predictions, targets)


class RecurrentModel(nn.Module):
    """ """

    def __init__(
        self,
        input_size,
        output_size,
        recurrent_size,
        hidden_size,
        n_layers=1,
        dropout=0.0,
        batch_first=True,
        dtype=torch.float32,
        device=None,
    ):
        nn.Module.__init__(self)
        self.input_size = input_size
        self.recurrent_size = recurrent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_first = batch_first
        self.rnn = nn.GRU(
            input_size,
            self.recurrent_size,
            self.n_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=self.bidirectional,
        )
        dense_in_features = self.recurrent_size
        self.dense = nn.Linear(
            in_features=dense_in_features,
            out_features=self.hidden_size,
        )
        self.output = nn.Linear(
            in_features=self.hidden_size,
            out_features=output_size,
        )

    def init_hidden(self, batch_size):
        n_layers = self.n_layers
        return torch.zeros(n_layers, batch_size, self.recurrent_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h0 = self.init_hidden(batch_size).type(x.type())
        output, h = self.rnn(x, h0)
        flatten_shape = self.recurrent_size
        dense = self.dense(output.contiguous().view(-1, flatten_shape))
        y = self.output(dense)
        y = y.view(batch_size, seq_len, self.output_size)

        return y


def prob_x_given_context(rnn, x, context, pitch_idxs, dur_idxs):
    # Assume that batch_size == 1, i.e., there is only one
    # sequence

    if isinstance(context, np.ndarray):
        if context.ndim == 2:
            context = context[np.newaxis, :, :]
        context = torch.tensor(context).to(rnn.dtype)
    softmax = nn.Softmax(dim=0)
    y = rnn(context)[-1, 1]
    pitch_prob = softmax(y[pitch_idxs]).detach().cpu().numpy()
    duration_prob = softmax(y[dur_idxs]).detach().cpu().numpy()

    x_pitch = x[pitch_idxs]
    x_dur = x[dur_idxs]
    pp = np.prod((pitch_prob) ** x_pitch * (1 - pitch_prob) ** (1 - x_pitch))
    dp = np.prod((duration_prob) ** x_dur * (1 - duration_prob) ** (1 - x_dur))
    p_x_given_contex = pp * dp
    return p_x_given_contex


def find_nearest(array, value):
    """
    From https://stackoverflow.com/a/26026189
    """
    idx = np.clip(np.searchsorted(array, value, side="left"), 0, len(array) - 1)
    idx = idx - (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
    return idx
