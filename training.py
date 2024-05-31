import random
import time

import numpy as np

from tqdm import tqdm

import wandb as wb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from typing import Optional, Tuple

from abc import ABC, abstractmethod

from dataset import Dataset

import os


class Trainer(ABC):
    def __init__(
            self,
            dataset: Dataset,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            model: nn.Module,
            optimizer: Optimizer,
            loss_fun: nn.Module,
            epochs: int
    ):
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self._lr = Trainer.get_lr(self.optimizer)
        self._epochs = epochs
        self._device = Trainer.get_target_device()
        self._global_train_step = 0
        self._global_val_step = 0
        self._global_test_step = 0
        self._current_epoch = 0

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, new_epoch):
        if isinstance(new_epoch, int) and new_epoch > 0:
            self._epochs = new_epoch
            return

        print("Please enter a valid epoch")

    def optimizing_predictor(
            self,
            out_path: str,
            adapt_lr_factor: Optional[float] = None,
            early_stopping: bool = False
    ) -> Tuple[float, float, float, float, float, float]:
        """Optimizes a given model for a number of epochs and saves the best model.

        The function computes both the defined loss and the accuracy between the output of the model and the given target.
        Depending on the best loss on the validation data, the best model is then saved to the specified file.
        Moreover, wandb is utilized in order to monitor the training process.
        Finally, a scheduling of the learning rate is implemented as well.

        Parameters
        ----------
        out_path: str
            Path to the file where the results and best model should be stored.
        adapt_lr_factor: float = None
            Factor used to adapt the learning rate if the model starts to over-fit on the training data.
        early_stopping: bool = False
            Bool used to specify if early stopping should be applied.

        Returns
        -------
        Tuple[float, float, float, float, float, float]
            A tuple containing the average train and validation loss/accuracy as well as the final test loss/accuracy.
        """

        best_loss = 0
        # Tell wandb to watch the model.
        wb.watch(self.model, criterion=self.loss_fun, log="all", log_freq=10)
        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []
        print("\nStarting to train Model")
        for epoch in range(self.epochs):

            train_loss, train_acc = self.train_model()
            val_loss, val_acc = self.eval_model()

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            validation_losses.append(val_loss)
            validation_accuracies.append(val_acc)

            wb.log(
                {"train/loss": train_loss,
                 "train/accuracy": train_acc,
                 "val/loss": val_loss,
                 "val/accuracy": val_acc,
                 "epoch": epoch}
            )

            print(f"\nEpoch: {str(epoch + 1).zfill(len(str(self.epochs)))} (lr={self._lr:.6f}) || "
                  f"Validation loss: {val_loss:.4f} || "
                  f"Validation accuracy: {val_acc:.4f} || "
                  f"Training loss: {train_loss:.4f} || "
                  f"Training accuracy: {train_acc}")

            # Check for early stopping.
            if early_stopping:
                if np.argmin(validation_losses) <= epoch - 5:
                    print(f"\nEarly stopping on epoch {epoch}!")
                    test_loss, test_acc = self.eval_model()
                    print(f"\nFinal loss: {test_loss}")
                    print("\nDone!")

                    return (np.mean(np.array(train_losses)).item(),
                            np.mean(np.array(train_accuracies)).item(),
                            np.mean(np.array(validation_losses)).item(),
                            np.mean(np.array(validation_accuracies)).item(),
                            test_loss,
                            test_acc)

            # Either save the best model or adapt the learning rate if necessary.
            if not best_loss or val_loss < best_loss:
                best_loss = val_loss
                torch.save({"epoch": epoch,
                            "model_dict": self.model.state_dict(),
                            "optimizer_dict": self.optimizer.state_dict(),
                            "best_val_loss": best_loss}, out_path)
                print(f"\nModel saved to {out_path}")
            else:
                if adapt_lr_factor is not None:
                    self._lr /= adapt_lr_factor
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self._lr
                    print(f"\nNew learning rate: {self._lr:.6f}")

            print("\n" + 100 * "=")

        test_loss, test_acc = self.eval_model(save_predictions=True)

        # Necessary to work with model in jupyter notebook after training is done.
        wb.unwatch(self.model)

        print(f"\nFinal loss: {test_loss}")
        print("\nDone!")

        return (np.mean(np.array(train_losses)).item(),
                np.mean(np.array(train_accuracies)).item(),
                np.mean(np.array(validation_losses)).item(),
                np.mean(np.array(validation_accuracies)).item(),
                test_loss,
                test_acc)

    def eval_model(self, save_predictions: bool = False) -> tuple[float, float]:
        """Evaluates a given model on test data.

        Parameters
        ----------
        save_predictions: bool = False
            Bool used to decide whether to log the model predictions or not.

        Returns
        -------
        float, float
            Returns the specified average loss and accuracy.
        """

        # Turn on evaluation mode for the model.
        self.model.eval()

        total_loss = []
        total_acc = []
        test_table = self.create_table() if save_predictions else None

        # Compute the loss with torch.no_grad() as gradients aren't used.
        with torch.no_grad():
            for idx, data, target in tqdm(self.test_loader, desc="Evaluating model on val/test set"):
                data, target = data.float().to(self._device), target.float().to(self._device)

                output, loss, acc = self.compute_loss_acc(data, target)

                # Log batch loss and accuracy as well as predictions.
                if save_predictions:
                    self.log_pred_target(test_table, idx, output, target)
                    wb.log({"test/batch loss": loss.item(), "test/batch accuracy": acc.item(),
                            "test/step": self._global_test_step})
                    self._global_test_step += 1
                else:
                    wb.log({"val/batch loss": loss.item(), "val/batch accuracy": acc.item(),
                            "val/step": self._global_val_step})
                    self._global_val_step += 1

                # Compute total loss.
                total_loss.append(loss.item())
                # Compute the total accuracy.
                total_acc.append(acc.item())

            # log final table
            if save_predictions:
                wb.log({"test/predictions": test_table})

        return np.mean(np.array(total_loss)).item(), np.mean(np.array(total_acc)).item()

    def train_model(self) -> tuple[float, float]:
        """Trains a given model on the training data.

        Returns
        -------
        float, float
            The specified average loss and accuracy.
        """

        # Put the model into train mode and enable gradients computation.
        self.model.train()
        torch.enable_grad()

        total_loss = []
        total_acc = []

        lr = Trainer.get_lr(self.optimizer)

        for _, data, target in tqdm(self.train_loader, desc=f"Training epoch {self._current_epoch + 1} ({lr=:.6f})"):
            data, target = data.float().to(self._device), target.float().to(self._device)

            output, loss, acc = self.compute_loss_acc(data, target)
            # Compute the gradients.
            loss.backward()
            # Perform the update.
            self.optimizer.step()
            # Reset the accumulated gradients.
            self.optimizer.zero_grad()
            # Log batch loss and accuracy.
            wb.log({"train/batch loss": loss.item(), "train/batch accuracy": acc.item(),
                    "train/step": self._global_train_step})
            self._global_train_step += 1
            # Compute the total loss.
            total_loss.append(loss.item())
            # Compute the total accuracy.
            total_acc.append(acc.item())

        self._current_epoch += 1

        return np.mean(np.array(total_loss)).item(), np.mean(np.array(total_acc)).item()

    @abstractmethod
    def compute_loss_acc(
            self,
            data: torch.Tensor,
            target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the loss and accuracy from the given data and target.

        Parameters
        ----------
        data : torch.Tensor
            Data used to calculate the model output and compute the loss as well as the accuracy.
        target: torch.Tensor
            Target used to compute the loss as well as the accuracy.

        Returns
        -------
        torch.Tensor, torch.Tensor, torch.Tensor
            The model output as well as the loss and accuracy for the respective data and target.
        """
        pass

    @abstractmethod
    def log_pred_target(self, test_table: wb.Table, idx: torch.Tensor, output: torch.Tensor, target: torch.Tensor):
        """Log the predictions and the respective target to the test table."""
        pass

    @abstractmethod
    def create_table(self) -> wb.Table:
        """Creates a table to log predictions"""
        pass

    @staticmethod
    def set_random_seed(seed: int | None = 42) -> int:
        """Decide if behaviour is random or set by a seed."""
        if seed is None:
            seed = time.time_ns() % (2 ** 32)

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")

        return seed

    @staticmethod
    def get_lr(optimizer):
        """Get the learning rate used for optimizing."""
        for param_group in optimizer.param_groups:
            return param_group['lr']

    @staticmethod
    def get_target_device():
        """Get the target device where training takes place."""
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
