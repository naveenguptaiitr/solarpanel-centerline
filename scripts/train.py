import os
import sys
sys.path.append("../")

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

from helper import *
from models import *
from dataset import SolarTrackerDataset

import segmentation_models_pytorch as smp

import torch
from torch.utils.data import DataLoader



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Solar Tracker Centerline Detection Model Training.')

    parser.add_argumen('--model_architecture', type=str, default='unet', help='Model architecture type to use (U-Net or Transformers)')
    parser.add_argument('--model_name', type=str, default='unet', help='Pre-trained model names (unet, unet-plus-plus, deeplabv3, transformers)')
    parser.add_argument('--encoder')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    





