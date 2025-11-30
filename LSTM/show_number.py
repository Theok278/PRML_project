#!/usr/bin/env python3

import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
import random
from pretreat_points import pretreat_points

number = "3"
nToPlot = 9
normalization = 1 # To normalize

files = sorted(glob.glob(f"../../digits_3d/training_data/stroke_{number}_*.csv"))
n = len(files)
print(f"Found {n} files for digit {number}.")
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
    pts = pretreat_points(df.to_numpy(), normalization)
    df = pd.DataFrame(pts, columns=["x", "y", "z"])

    # Plot
    ax.scatter(df["x"], df["y"], df["z"], color='blue', s=5)
    ax.scatter(df["x"][0], df["y"][0], df["z"][0], color='red', s=30)
    
    ax.view_init(elev=90, azim=-90)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.title(nums[i])

plt.show()
