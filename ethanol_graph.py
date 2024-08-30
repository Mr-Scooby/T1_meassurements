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

def methanol_file_to_df(file_path): 

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

if __name__ == "__main__":

    df =  methanol_file_to_df("ethanol_data.txt")
    print(df.columns)
    col = df.columns
    plt.plot(df[col[0]], df[col[1]])
    plt.plot(df[col[0]], df[col[2]])
    plt.plot(df[col[0]], df[col[3]])
    plt.show()
