#!/usr/bin/env python3

import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
import random
from pretreat_points import pretreat_points

number = "8"
nToPlot = 100
orientate = 1 # To project on maximum variations PCs
normalization = 1 # To normalize
rectification_orientation = 1 # To orientate so beginning of drawing is on top of the image

files = sorted(glob.glob(f"digits_3d/training_data/stroke_{number}_*.csv"))
n = len(files)
nums = sorted(random.sample(range(n), nToPlot))

cols = math.ceil(math.sqrt(nToPlot))
rows = math.ceil(nToPlot / cols)

fig = plt.figure()    

for i in range(nToPlot):
    # Load
    f = files[nums[i]]
    df = pd.read_csv(f, header=None, names=["x", "y", "z"])
    ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    
    # Pretreatment
    pts = pretreat_points(df.to_numpy(), orientate, normalization, rectification_orientation)
    df = pd.DataFrame(pts, columns=["x", "y", "z"])

    # Plot
    ax.scatter(df["x"], df["y"], df["z"], color='blue', s=5)
    ax.scatter(df["x"][0], df["y"][0], df["z"][0], color='red', s=30)
    ax.view_init(elev=-90, azim=0)
    plt.title(nums[i])

plt.show()
