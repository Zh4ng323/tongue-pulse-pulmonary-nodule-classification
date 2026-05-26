# -*- coding: utf-8 -*-
"""
Global configuration for pulmonary nodule classification analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import random

# =============================================================================
# Paths
# =============================================================================

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "sample_data.csv")
OUTPUT_BASE_DIR = os.path.join(_PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# =============================================================================
# Data
# =============================================================================

TARGET_COLUMN = "Group"
TARGET_LABELS = {0: "Benign", 1: "Malignant"}

# =============================================================================
# Reproducibility
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# Features
# =============================================================================

TONGUE_FEATURES = [
    'Per-all', 'Per-part',
    'TB-CON', 'TC-CON', 'TB-ASM', 'TC-ASM',
    'TB-ENT', 'TC-ENT', 'TB-MEAN', 'TC-MEAN',
    'TB-B', 'TB-R', 'TB-G', 'TC-R', 'TC-G', 'TC-B',
    'TB-H', 'TB-I', 'TB-S', 'TC-H', 'TC-I', 'TC-S',
    'TB-L', 'TB-a', 'TB-b', 'TC-L', 'TC-a', 'TC-b',
    'TB-Y', 'TB-Cr', 'TB-Cb', 'TC-Y', 'TC-Cr', 'TC-Cb'
]

PULSE_FEATURES = [
    'h1', 'h3', 'h4', 'h5',
    't', 't1', 't4', 't5',
    'h3/h1', 'h1/t1', 'h4/h1',
    't1/t', 't4/t5',
    'w1/t', 'w2/t'
]

# =============================================================================
# Fonts
# =============================================================================

FONT_CHINESE = ['Microsoft YaHei', 'SimHei', 'sans-serif']
FONT_ENGLISH = 'Times New Roman'

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# =============================================================================
# Global settings
# =============================================================================

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = FONT_CHINESE
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")

# =============================================================================
# Thresholds
# =============================================================================

Q_THR = 0.05                # FDR-corrected p-value threshold
DIFF_Q_THR = 0.05           # Differential significance threshold

OVERVIEW_R_THR = 0.15       # Correlation heatmap annotation threshold
BACKBONE_R_THR = 0.20       # Network graph edge threshold

DIFF_DR_THR = 0.20          # Differential network |dr| threshold
DIFF_REQUIRE_MAXR = 0.25    # Minimum max(r) for differential edges

CCA_R_LINE = 0.30           # Canonical correlation threshold
STRONG_R_THR = 0.50         # Strong correlation threshold

# Network graph visualization
NET_NODE_MIN_SIZE = 300
NET_NODE_MAX_SIZE = 2500
NET_EDGE_MIN_WIDTH = 0.5
NET_EDGE_MAX_WIDTH = 6.0

NET_POSITIVE_COLOR = '#009688'
NET_NEGATIVE_COLOR = '#FF4500'
NET_TONGUE_NODE_COLOR = '#6B8E23'
NET_PULSE_NODE_COLOR = '#4682B4'

# =============================================================================
# Version
# =============================================================================

VERSION = "2.0"
VERSION_DATE = "2026-05-27"
AUTHORS = ["Zhang Guohao", "Shi Yulin"]
