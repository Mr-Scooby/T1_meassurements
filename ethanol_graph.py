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
    logger.debug(f"data dict = \n" + "\n".join("{}\t{}".format(k, v) for k, v in data_dict.items()))
    return pd.DataFrame(data_dict)

def plot_integral(df):
 # Total number of columns
    # Total number of columns
    total_columns = len(df.columns)
    
    # Number of groups
    num_groups = total_columns // 4
    
    # Determine number of rows needed for 2 columns
    n_cols = 3
    n_rows = (num_groups + n_cols - 1) // n_cols  # Calculate rows needed

    # Create a figure with subplots arranged in a grid layout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3* n_rows), sharex=False)

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    # Loop through each group and plot the data
    for i in range(num_groups):
        # Calculate the column indices for this group
        base_idx = i * 4
        x_col = df.columns[base_idx]
        oh_col = df.columns[base_idx + 1]
        c21_col = df.columns[base_idx + 2]
        c31_col = df.columns[base_idx + 3]
        
        ax = axes[i]
        ax.plot(df[x_col], df[oh_col], label=f'OH')
        ax.plot(df[x_col], df[c21_col], label=f'C2')
        ax.plot(df[x_col], df[c31_col], label=f'C3')
    
        ax.set_title(f'{x_col}')
        ax.set_ylabel('Values')
        #ax.set_title(f'Plots for Group {i+1}')
        ax.legend()
    # Turn off any unused subplots
    for j in range(num_groups, len(axes)):
        axes[j].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":

    df =  ethanol_file_to_df("ethanol_data.txt")
    print(df.columns)
    print(len(df.columns)/4)
    col = df.columns
    
    plot_integral(df)
   
