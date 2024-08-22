#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd

plt.style.use('./graphs_style.mplstyle')



data = pd.read_csv("datos.csv")

media = data.groupby('concentracion').mean().reset_index()

print(data.sort_values("concentracion"))

fig = plt.figure(figsize=(12,6))
#plt.scatter(data["concentracion"], data["T1"])
plt.plot(media["concentracion"], media["T1"], "*--")
plt.grid()
plt.ylabel(r"$T_1$ [s]")
plt.xlabel(r"%$H_2O$ ")
plt.xticks(media["concentracion"])
plt.title("Data from SpinSolver", fontsize="xx-large") 
plt.show()


