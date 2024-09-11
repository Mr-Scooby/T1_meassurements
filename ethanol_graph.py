#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import extract_data as xd
from scipy.optimize import curve_fit
import pandas as pd
import re
from inspect import signature

plt.style.use('./graphs_style.mplstyle')

logger = logging.getLogger(__name__)

def setup_log():
    logger.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)


def extract_etha_percentage(header):
    """Extracts the ETHA percentage from a header string."""
    match = re.search(r'\((\d+(?:\.\d+)?)%ETHA', header)
    if match:
        return float(match.group(1))
    return 0  # Fallback if no match is found

REGEX_ETHANOL = r'(\d{6}-\d{6} T1 \(\d+(?:\.\d+)?%ETHA\d+(?:\.\d+)?%D2O).*'

def ethanol_file_to_df(file_path): 

    logger.info("converting file to DF")
    sections = xd.section_split(file_path, regex= REGEX_ETHANOL)
    logger.info("Ethanol_data_extraction: Section headers: \n" + "\n".join(sec for sec in sections[::2]))
    sorted_sections = xd.sort_sections(sections,extract_etha_percentage )
    # Dictionary to store the extracted data
    data_dict = {}
    # Process each section
    for idx, header in enumerate(sorted_sections):
        if idx % 2 == 0:  # Ensure we're processing headers
            
            data_section = sorted_sections[idx + 1].strip()
            
            logger.info(f"Extracting data from section: {header}")
            data_extracted = xd.process_section(data_section, column_indices=[1,2,4,6])
            data_x = data_extracted[0]
            data_OH = data_extracted[1]
            data_C2 = data_extracted[2]
            data_C3 = data_extracted[3]

            column_name = xd.column_name_formatter(header, label="ETHA")
 
            data_dict[column_name + "_x"] = data_x    
            data_dict[column_name + "_OH"] = data_OH    
            data_dict[column_name + "_C2"] = data_C2    
            data_dict[column_name + "_C3"] = data_C3    
    ## Create a DataFrame from the dictionary
    logger.debug(f"ethanol_file_to_df: data dict = \n" + "\n".join("{}\t{}".format(k, v) for k, v in data_dict.items()))
    return pd.DataFrame(data_dict)

def plot_integral(df):
    # Total number of columns
    total_columns = len(df.columns)
    
    # Number of groups
    num_groups = total_columns // 4
    logger.info(f"plot_integral: number of subplots = {num_groups}")
    
    # Initialize figure count
    fig_count = 0
    
    for i in range(num_groups):
        if i % 9 == 0:
            fig_count += 1
            n_cols = 3
            n_rows = min(3, (num_groups - i + n_cols - 1) // n_cols)  # Calculate rows needed
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=False)
            axes = axes.flatten()

        # Calculate the column indices for this group
        base_idx = i * 4
        x_col = df.columns[base_idx]
        oh_col = df.columns[base_idx + 1]
        c21_col = df.columns[base_idx + 2]
        c31_col = df.columns[base_idx + 3]

        ax = axes[i % 9]
        ax.plot(df[x_col], df[oh_col], "*--", label=f'OH' )
        ax.plot(df[x_col], df[c21_col],"*--", label=f'C2')
        ax.plot(df[x_col], df[c31_col],"*--", label=f'C3')

        ax.set_title(f'{x_col.strip("_x")}')
        #ax.set_ylabel('Values')
        ax.legend()
        ax.grid(":")
        ax.set_yticks([df[oh_col].iloc[0], 0, df[oh_col].iloc[-1], df[c21_col].iloc[-1], df[c31_col].iloc[-1] ])
        plt.tight_layout()
    # Turn off any unused subplots in the last figure
    g = num_groups % 9 
    for j in range(g , len(axes)):
        if g == 0:
            continue
        else: 
            axes[j].axis('off')
    
    # Adjust layout for all figures
    
def T1_calculation(x, y, p0 = [1,1]):
    # Define the fitting function
    f = lambda t, a, b: a * (1 - 2 * np.exp(-t /  b))
    
    # Perform curve fitting
    popt, pcov = curve_fit(f, x, y, p0= p0 )
    logger.debug(f"popt output = {popt}")
    logger.debug(f"pcov output = {pcov}")
    
    # Extract the parameters and their errors
    a, b = popt
    a_error, b_error = np.sqrt(np.diag(pcov))
    
    return (a, b, a_error, b_error)

def ethanol_t1_calc(df):
    # Initialize an empty dictionary to hold all the data
    data = {
        "sample": [],
        "OH_T1": [],
        "OH_T1_error": [],
        "OH_a": [],
        "OH_a_error": [],
        "C2_T1": [],
        "C2_T1_error": [],
        "C2_a": [],
        "C2_a_error": [],
        "C3_T1": [],
        "C3_T1_error": [],
        "C3_a": [],
        "C3_a_error": []
    }

    # Number of groups
    num_groups = len(df.columns) // 4

    # Loop through each group and calculate T1 values, a, and errors
    for i in range(num_groups):
        # Calculate the column indices for this group
        base_idx = i * 4
        x_col = df.columns[base_idx]
        oh_col = df.columns[base_idx + 1]
        c2_col = df.columns[base_idx + 2]
        c3_col = df.columns[base_idx + 3]

        # Perform T1, a, and error calculations
        oh_a, oh_t1, oh_a_error, oh_t1_error = T1_calculation(df[x_col], df[oh_col], p0 = [df[oh_col].iloc[-1], 1 ])
        c2_a, c2_t1, c2_a_error, c2_t1_error = T1_calculation(df[x_col], df[c2_col], p0 = [df[c2_col].iloc[-1], 1 ])
        c3_a, c3_t1, c3_a_error, c3_t1_error = T1_calculation(df[x_col], df[c3_col], p0 = [df[c3_col].iloc[-1], 1 ])

        # Append the results to the corresponding lists in the dictionary
        sample_name = x_col.strip('_x')
        data["sample"].append(sample_name)
        data["OH_T1"].append(oh_t1)
        data["OH_T1_error"].append(oh_t1_error)
        data["OH_a"].append(oh_a)
        data["OH_a_error"].append(oh_a_error)
        data["C2_T1"].append(c2_t1)
        data["C2_T1_error"].append(c2_t1_error)
        data["C2_a"].append(c2_a)
        data["C2_a_error"].append(c2_a_error)
        data["C3_T1"].append(c3_t1)
        data["C3_T1_error"].append(c3_t1_error)
        data["C3_a"].append(c3_a)
        data["C3_a_error"].append(c3_a_error)

    # Extract percentage from sample names and create the DataFrame
    percentage = lambda name: re.search(r'(\d+(?:\.\d+)?)ETHA', name)[0].strip('ETHA')
    t1_df = pd.DataFrame(data)
    t1_df.index = [percentage(name) for name in data["sample"]]

    logger.info(f"T1 values and a values:\n {t1_df}")
    return t1_df

def integral_plot(df, change_xaxis = True, fitting = False):

    if change_xaxis == True:
       x_range = [float(idx) for idx in df.index]
    else: 
        x_range = df.index
    
    groups = ["OH", "C2", "C3"]
    colors = "brg"
    func = lambda x,a,b : a *x + b 

    fig, ax = plt.subplots(3,1, figsize=(12,6), sharex = True, sharey = False ) 
    for idx, group in enumerate(groups):
        ax[idx].errorbar(x_range, df[group + "_a"], yerr=df[group + "_a_error"], fmt ="*", label = group , color= colors[idx])
        ax[idx].grid(":")
        ax[idx].legend()
        if fitting == True: 
            add_fit(df[group + "_a"], func, ax[idx], color=colors[idx])
    plt.xlabel("%ETHA")



def T1_plot(t1_valeus, change_xaxis= True, fitting = False):

    fig, ax  = plt.subplots(1,1, figsize=(12,5))
    x_range = t1_valeus.index
    
    groups = ["OH", "C2", "C3"]
    color = "brg"

    fit_func = lambda x,a,b,d : a * np.exp(-  b * x ) + d
    #fit_func = lambda x,a,b : 1 / ( a * x + b) 
    if change_xaxis == True: 
        x_range = [float(idx) for idx in t1_valeus.index]
        ax.set_xlim([0, max(x_range) * 1.05 ])

    for idx, group in enumerate(groups):
        ax.errorbar(x_range, t1_valeus[group + "_T1"], yerr=t1_valeus[group + "_T1_error"], fmt = "*",  label= group, color  = color[idx])

        if fitting == True: 
            add_fit(t1_valeus[group + "_T1"], fit_func, ax, color[idx])

    ax.grid(":")
    ax.legend()
    ax.set_xlabel("% Etha")
    ax.set_ylabel("T1 [s]")
    plt.tight_layout()

def add_fit(df, fit_func , ax, color = "b"):
    """
    Adds the fitting curve to the plot. 
    Args: 
        df: DataFrame: with the values to fit
        ax: Fig.axes: axes of the plot to insert the fitting curve
    returns:
        ax: with the fitting inserted.
    """
    
    # Extract x and y data from df
    x_range = np.array([float(idx) for idx in df.index])
    y_data = df.values
    
    # Determine the number of parameters in fit_func using inspect.signature
    num_params = len(signature(fit_func).parameters) - 1  # Exclude the first parameter `x`
    
    # Generate initial guess (p0) based on the number of parameters
    # Here, we simply initialize all parameters as 1.0, except the first and last point.
    p0 = [df.values[0]] + [0.3] * (num_params - 2) + [df.values[-1]]

    # Perform the curve fitting
    popt, pcov = curve_fit(fit_func, x_range, y_data, p0=p0)
    
    # Log the results
    logger.debug(f"popt output = {popt}")
    logger.debug(f"pcov output = {pcov}")
    
    # Generate the x values for plotting the fit
    x = np.linspace(x_range.min(), x_range.max(), 500)
    
    # Plot the fit on the axes with the fitted parameters
    ax.plot(x, fit_func(x, *popt), color=color)
    
    return ax



def plt_fitting_2(sample_idxs, fitting_df, data_df ):
    """
    Plots the fitting for the data recorded in Mnova for the intergals.
    args: 
        smaple_idxs: int ot list: provides the indices of the samples to make the fitting plot
        fitting_df: DataFrame: With the smaples ids and the fitting parameters to make the plot.
        data_df: DataFrame: With the data extracted from the file

    retruns:
        figure
    """
    # Loc samples names 
    samples = fitting_df.loc[sample_idxs, "sample" ]
    # Ensure samples is always a list, even if it's a single string
    if isinstance(samples, pd.Series):
        samples = samples.tolist()  # Convert to list if it's a Series
    else:
        samples = [samples]  # Wrap the single string in a list
    logger.info(f"plt_fitting_2: samples = {samples}")

    fig, axes = plt.subplots(len(samples), 1 , figsize=(12, 8 ), sharex=False)
    # Ensure axes is always treated as an array, even for a single subplot
    axes = np.atleast_1d(axes)
    axes = axes.flatten()

    # Define the additional y-series columns to scatter for each sample
    y_series = ["OH", "C2", "C3"]  # Add more y-series if needed

    # fitting function:
    fitting = lambda M, T1, t  : M * ( 1 - 2 * np.exp( - t / T1 ) ) 

    # Changing index to sample column
    df_indexed = fitting_df.set_index("sample")
    # Plot for each sample
    for i, sample in enumerate(samples):
        logger.debug(f"plt_fitting_2: loop: sample = {sample}")
        x_data = data_df[sample + "_x"]
        
        t = np.linspace(0, x_data.iloc[-1])
        for y in y_series:
            # Scatter multiple y-series for each sample on the same subplot
            axes[i].scatter(x_data, data_df[sample + "_"+y], label= y, marker="x")

            M , T1 = df_indexed.loc[sample, [y+"_a", y+"_T1" ]]
            axes[i].plot(t, fitting(M, T1, t)  )
            axes[i].set_title(sample) 

        # Add legend to the plot
        axes[i].legend(loc = "lower right")
        axes[i].grid(":")




if __name__ == "__main__":

    setup_log()
    # Extracting data from file
    file = input("File number : ")
    df =  ethanol_file_to_df(f"ethanol_data_{file}.txt")
    # Calculating fitting to T1 Curve from data.
    fitting_df = ethanol_t1_calc(df)
    
    # Filtering df to obtain T1 Values and Total magnetization from fitting calculations.
    magentization_df = fitting_df.filter(regex="^sample$|_a.*", axis=1)
    logger.info(f"data frame filtered with magentization:\n {magentization_df}")

    #T1_df = fitting_df.filter(regex="^(?!.*_a).*", axis=1)
    #logger.info(f"data frame filtered with T1 values :\n {T1_df}")
    #plt_fitting([2,3,4], fitting_df, df )
    #integral_plot(fitting_df, True, True )
    #T1_plot(fitting_df, change_xaxis= True , fitting = True)
    
    #plot_integral(df)
    
    #plt_fitting_2(["02"], fitting_df, df )
    #plt.show()
    #logger.info(fitting_df["OH_T1"])#.iloc[:,1])

    T1_plot(fitting_df.loc["01":"80"], fitting= True)
    plt.show()


#    £samples = fitting_df.loc["01", "sample" ] # Loc sample name from idx. 0 is the sample column
#ogger.info([ i for i in samples])


