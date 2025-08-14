import os
import tempfile
from functools import partial
import datetime
import wandb
import matplotlib.pyplot as plt
import torch
import pathlib
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from data import WeatherBenchModule
from models import LightningDRN, EmbeddingMLP


if __name__ == "__main__":
    print(torch.cuda.is_available())