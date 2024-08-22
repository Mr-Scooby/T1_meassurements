#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# Codigo para hacer el fitting con los datos de Mnova. 


import numpy as np 
import matplotlib.pyplot as plt 
import logging
from scipy.optimize import curve_fit
import pandas as pd

plt.style.use('./graphs_style.mplstyle')
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

y = np.array([-8.98122
,-8.63694 
,-8.11342 
,-7.07844 
,-5.17454 
,-1.89234 
,2.76553 
,7.22691 
,9.24916 
,9.4732])

x = np.geomspace(0.1, 60, 10 )

fig = plt.figure(figsize=(12,6))
plt.plot(x,y,"-*", label="Data" ) 
plt.grid()
plt.ylabel(r"M")
plt.xlabel(r"t [s]")
#plt.show()

f = lambda t,a, b,c : a - a* np.exp(-t/b + c) 
# Fit the function a * np.exp(b * t) + c to x and y
popt, pcov = curve_fit(f , x, y)
logger.debug(f"popt output = {popt}")
logger.debug(f"pcov output = {pcov}")


a = popt[0]
b = popt[1]
c = popt[2]

x_1 = np.geomspace(0.1, 60, 500 )
y_fitted =  a-  a*np.exp(-x_1/b+ c)
plt.plot(x_1,y_fitted, label="fitting") 
plt.text(30,5,r"$ f(t) = A - Ae^{-t/T_1 + c}$", size = 20) 
plt.text(30,0,f"A = {a}\nT_1 = {b}\nC={c}", size = 20) 
logger.info(f"T1 meassure : {b}")
plt.legend()
#plt.show()


data = pd.read_xml("mnova_fit_90H2O.xml")
print(data)

