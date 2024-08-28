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

def T_1 (x, y): 
    f = lambda t,a,b : a* ( 1 -   2 * np.exp(-t / b ))
    # Fit the function a * np.exp(b * t) + c to x and y
    popt, pcov = curve_fit(f , x, y)
    logger.debug(f"popt output = {popt}")
    logger.debug(f"pcov output = {pcov}")
    
    return (popt[1], np.sqrt(np.diag(pcov))[1])

if __name__ == "__main__":

    FILE_NAME = "processed_data_3.txt"
    
    data = xd.file_to_DF(FILE_NAME) 
    logger.info(data)
    
    x_axis = data.filter(regex="_x", axis=1)
    y_axis = data.filter(regex="^(?!.*_x$).*", axis=1)
    
    logger.info(f"x_axis data:\n{x_axis}") 
    logger.info(f"y_axis data:\n{y_axis}") 
    
    
    logger.info("Calculating T_1") 
    T_1_values={column:  T_1(x_axis[column+"_x"], y_axis[column])[0] for column in y_axis.columns}
    T_1_errors={column:  T_1(x_axis[column+"_x"], y_axis[column])[1] for column in y_axis.columns}
    
    indexs = [int(re.search(r'(\d+)H2O',key ).group(1)) for key in  T_1_values.keys()]
    logger.debug(indexs)
    
    T_1 = pd.DataFrame(T_1_values.values(), index=indexs, columns=["T1"]).sort_index()
    T_1["error"] = T_1_errors.values()
    logger.info(f"T_1 Values:\n{T_1}") 
    
    
    fig = plt.figure(1, figsize=(12,6))
    plt.errorbar(T_1.index, T_1["T1"], T_1["error"],fmt="." )
    plt.grid()
    plt.ylabel(r"$T_1$ [s]")
    plt.xlabel(r"%$H_2O$ ")
#    plt.xticks(T_1.index)
    plt.title("T_1 fit from python", fontsize="xx-large") 
    plt.show()
    
#    T1_mediado = T_1.groupby(T_1.index).mean()
#    logger.info(f"T_1 Values average:\n{T_1}") 
#    
#    fig = plt.figure(2, figsize=(12,6))
#    plt.plot(T1_mediado,"*--")
#    plt.grid()
#    plt.ylabel(r"$T_1 averge $ [s]")
#    plt.xlabel(r"%$H_2O$ ")
#    plt.title("T_1 fit from python, avergae", fontsize="xx-large") 
#    plt.show()
