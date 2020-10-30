#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils

# --------------- Setup options ---------------
is_sim = True  # Run in simulation?
# Directory containing 3D mesh files (.obj) of objects to be added to simulation
obj_mesh_dir = '/home/song/visual-pushing-grasping/objects/blocks' if is_sim else None
# Number of objects to add to simulation
num_obj = 10 if is_sim else None
tcp_host_ip = None
tcp_port = None
rtc_host_ip = None
rtc_port = None
if is_sim:
    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    workspace_limits = np.asarray(
        [[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
else:
    workspace_limits = np.asarray(
        [[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]])
is_testing = True
test_preset_cases = None
test_preset_file = None


robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
              tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
              is_testing, test_preset_cases, test_preset_file)
robot.check_sim()
