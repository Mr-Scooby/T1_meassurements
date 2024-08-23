#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import logging
import extract_data as xd 
import grafica_datos_procesados as gf 
from scipy.optimize import curve_fit

plt.rcParams.update({'font.family':'Times New Roman', 'mathtext.fontset':'cm', 'font.size':13})
# plt.style.use('./presentation.mplstyle')

FILE_NAME = "processed_data_3.txt"
data = xd.file_to_DF(FILE_NAME) 

x_axis = data.filter(regex="_x", axis=1)
y_axis = data.filter(regex="^(?!.*_x$).*", axis=1)

print(f"x_axis columns = {x_axis.columns}")
print(f"y_axis columns = {y_axis.columns}")
columns = data.columns
print(f"columns = {columns}")


f = lambda t,a, c : a * (1-  2 * np.exp(-t/c ))
# Fit the function a * np.exp(b * t) + c to x and y
popt, pcov = curve_fit(f ,x_axis["16H2O_240821-112704_x"], y_axis["16H2O_240821-112704"])


a = popt[0]
#b = popt[1]
c = popt[1]

print(f" T_1 = {c}")


plt.plot(x_axis["16H2O_240821-112704_x"], y_axis["16H2O_240821-112704"], "*")
x = np.linspace(0.1, 90, 1000)
plt.plot(x, f(x, a,c), color="red")
plt.show()

         
