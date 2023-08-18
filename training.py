import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer  # , lr_scheduler

import dill as pkl

from typing import Optional, Tuple

import os


def optimizing_predictor(
        train_loader: DataLoader,
        validation_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module,
        epochs: int,
        loss_function: nn.Module,
        optimizer: Optimizer,
        target_directory: str,
        adapt_lr_factor: Optional[float] = None,
        early_stopping: bool = False
) -> Tuple[list, list, float]:
    """Optimizes a given model for a number of epochs and saves the best model.

    The function computes both the defined loss between the output of the model and the given target.
    Depending on the best loss on the validation data, the best model is then saved to the specified file.
    Moreover, tensorboard is utilized in order to monitor the training process.
    Finally, a scheduling of the learning rate is implemented as well.

    Parameters
    ----------
    train_loader: DataLoader
        Data for training the model.
    validation_loader: DataLoader
        Data for monitoring validation loss.
    test_loader: DataLoader
        Data for checking how well the model works on unseen data.
    model: nn.Module
        Model to be trained.
    epochs: int
        Number of epochs to train the model for.
    loss_function: nn.Module
        Loss function used to compute the loss between the output of the model and the target.
    optimizer: Optimizer
        Specified optimizer to be used to optimize the model.
    target_directory: str
        Path to the directory where the results and best model should be stored.
    adapt_lr_factor: float = None
        Factor used to adapt the learning rate if the model starts to over-fit on the training data.
    early_stopping: bool = False
        Bool used to specify if early stopping should be applied.

    Returns
    -------
    Tuple[list, list, float]
        A tuple containing all train and validation losses as well as the final test loss.
    """

    best_loss = 0
    lr = get_lr(optimizer)
    writer = SummaryWriter(log_dir=target_directory)
    train_losses = []
    validation_losses = []
    print("\nStarting to train Model")
    for epoch in range(epochs):

        train_loss = train_model(model, optimizer, train_loader, loss_function, epoch)
        validation_loss = eval_model(model, validation_loader, loss_function)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        writer.add_scalar(tag="Loss/train",
                          scalar_value=train_loss,
                          global_step=epoch)
        writer.add_scalar(tag="Loss/test",
                          scalar_value=validation_loss,
                          global_step=epoch)

        print(f"\nEpoch: {str(epoch + 1).zfill(len(str(epochs)))} (lr={lr:.6f} || "
              f"Validation loss: {validation_loss:.4f} || "
              f"Training loss: {train_loss:.4f}")

        # Check for early stopping.
        if early_stopping:
            if np.argmin(validation_losses) <= epoch - 5:
                print(f"\nEarly stopping on epoch {epoch}!")
                test_loss = eval_model(model, test_loader, loss_function)
                writer.close()
                print(f"\nFinal loss: {test_loss}")
                print("\nDone!")

                return train_losses, validation_losses, test_loss

        # Either save the best model or adapt the learning rate if necessary.
        if adapt_lr_factor is not None:
            if not best_loss or validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(model, os.path.join(target_directory, "best_model.pt"))
                print("\nModel saved to best_model.pt")
            else:
                lr /= adapt_lr_factor
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                print(f"\nNew learning rate: {lr:.6f}")

        print("\n" + 100 * "=")

    writer.close()

    test_loss = eval_model(model, test_loader, loss_function)

    print(f"\nFinal loss: {test_loss}")
    print("\nDone!")

    return train_losses, validation_losses, test_loss


def eval_model(
        model: nn.Module,
        test_loader: DataLoader,
        loss_function: nn.Module,
        save_predictions: bool = False
) -> float:
    """Evaluates a given model on test data.

    Parameters
    ----------
    model: nn.Module
        Model used for evaluation.
    test_loader: DataLoader
        Data used for testing the model.
    loss_function: nn.Module
        Loss function used to determine the "goodness" of the model.
    save_predictions: bool = False
        Bool used to decide whether to return the model predictions or not.

    Returns
    -------
    float
        Returns the specified loss.
    """

    target_device = get_target_device()

    # Turn on evaluation mode for the model.
    model.eval()

    total_loss = []
    predictions = []

    # Compute the loss with torch.no_grad() as gradients aren't used.
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.float().to(target_device), target.long().to(target_device)

            outputs = model(data)
            predictions.append(outputs)

            # Compute the loss.
            loss = loss_function(outputs, target)

            # Compute total loss.
            total_loss.append(loss.item())

        # Save predictions if save predictions.
        if save_predictions:
            with open("predictions.pkl", "wb") as f:
                pkl.dump(predictions, f)

    return np.mean(np.array(total_loss)).item()


def train_model(
        model: nn.Module,
        optimizer: Optimizer,
        training_loader: DataLoader,
        loss_function: nn.Module,
        epoch: int
) -> float:
    """Trains a given model on the training data.

    Parameters
    ----------
    model: ImagePixelPredictor
        Model to be trained.
    optimizer: Optimizer
        Specified optimizer to be used to optimize the model.
    training_loader: DataLoader
        Data used for training the model.
    loss_function: nn.Module
        Loss function used to compute the loss between the output of the model and the given target.
    epoch: int
        Number of iteration.

    Returns
    -------
    float
        Returns the specified loss.
    """

    target_device = get_target_device()

    # Put the model into train mode and enable gradients computation.
    model.train()
    torch.enable_grad()

    total_loss = []

    lr = get_lr(optimizer)

    for data, target in tqdm(training_loader, desc=f"Training epoch {epoch + 1} "
                                                   f"(lr={lr:.6f})"):

        data, target = data.float().to(target_device), target.long().to(target_device)

        outputs = model(data)

        # Compute loss.
        loss = loss_function(outputs, target)

        # Compute the gradients.
        loss.backward()

        # Perform the update.
        optimizer.step()

        # Reset the accumulated gradients.
        optimizer.zero_grad()

        # Compute the total loss.
        total_loss.append(loss.item())

    return np.mean(np.array(total_loss)).item()


def get_lr(optimizer):
    """Get the learning rate used for optimizing."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_target_device():
    """Get the target device where training takes place."""
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
