#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import logging
import extract_data as xd
from scipy.optimize import curve_fit
import pandas as pd
import re

plt.style.use('./graphs_style.mplstyle')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


FILE_NAME = "processed_data_2.txt"

data = xd.file_to_DF(FILE_NAME) 
logger.info(data)

x_axis = data.filter(regex="_x", axis=1)
y_axis = data.filter(regex="^(?!.*_x$).*", axis=1)

logger.info(f"x_axis data:\n{x_axis}") 
logger.info(f"y_axis data:\n{y_axis}") 


def T_1 (x, y): 
    f = lambda t,a, b,c : a - a* np.exp(-t/b + c) 
    # Fit the function a * np.exp(b * t) + c to x and y
    popt, pcov = curve_fit(f , x, y)
    logger.debug(f"popt output = {popt}")
    logger.debug(f"pcov output = {pcov}")
    
    
    a = popt[0]
    b = popt[1]
    c = popt[2]

    return b


logger.info("Calculating T_1") 
T_1_values={column:  T_1(x_axis[column+"_x"], y_axis[column]) for column in y_axis.columns}

indexs = [int(re.search(r'(\d+)H2O',key ).group(1)) for key in  T_1_values.keys()]
logger.debug(indexs)

T_1 = pd.DataFrame(T_1_values.values(), index=indexs, columns=["T1"]).sort_index()
logger.info(f"T_1 Values:\n{T_1}") 


fig = plt.figure(1, figsize=(12,6))
plt.plot(T_1,"*")
plt.grid()
plt.ylabel(r"$T_1$ [s]")
plt.xlabel(r"%$H_2O$ ")
plt.xticks(T_1.index)
plt.title("T_1 fit from python", fontsize="xx-large") 
#plt.show()

T1_mediado = T_1.groupby(T_1.index).mean()
logger.info(f"T_1 Values average:\n{T_1}") 

fig = plt.figure(2, figsize=(12,6))
plt.plot(T1_mediado,"*--")
plt.grid()
plt.ylabel(r"$T_1 averge $ [s]")
plt.xlabel(r"%$H_2O$ ")
plt.title("T_1 fit from python, avergae", fontsize="xx-large") 
plt.show()

