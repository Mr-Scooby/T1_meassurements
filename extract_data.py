#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import re
import logging 
import matplotlib.pyplot as plt 

plt.style.use('./graphs_style.mplstyle')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)


def process_section(data_section, column_indices=[1,2]):
    """
    Extracts data from specified columns in a data section.

    Args:
        data_section (str): The section of the file containing the data.
        header (str): The header of the section being processed.
        column_indices (list): List of integers representing the column indices to extract.

    Returns:
        list: A list of lists, where each inner list contains the data from a specified column.
    """
    lines = data_section.strip().split('\n')
    logger.debug(f"process_section:lines = {lines}")

    # Initialize lists to store extracted data for each column
    extracted_data = [[] for _ in column_indices]

    # Process each line and extract data from specified columns
    for line in lines[2:12]:  # Extract only the first 10 data points after skipping the header
        columns = line.split()
        logger.debug(f"process_section: columns = {columns}")
        
        # Extract data from specified columns
        for i, col_idx in enumerate(column_indices):
            if col_idx < len(columns):
                extracted_data[i].append(float(columns[col_idx]))
            else:
                logger.warning(f"Column index {col_idx} is out of range for line: {line}")

    logger.debug(f"process_section: extracted_data = {extracted_data}")
    return extracted_data

def extract_h2o_percentage(header):
    """Extracts the H2O percentage from a header string."""
    match = re.search(r'\((\d+)%H2O', header)
    if match:
        return int(match.group(1))
    return 0  # Fallback if no match is found

def section_split(file_path, regex=r'(\d{6}-\d{6} T1 \(\d+%H2O\d+%D2O).*'):
    """Splits the file into sections based on regex"""
    # Read the file content
    with open(file_path, 'r') as file:
        logger.info(f"reading file {file_path}")
        content = file.read()

    # Split the content by header lines
    sections = re.split(regex, content)[1:]
    logger.info(f"section_split: Number of sections : {len(sections)//2}")

    return sections

def sort_sections(sections, sort_key_fucntion):
    """
    Sorts the sections based on the section header using the provided sort_key_function.
    
     Args:
        sections (list): A list where even indices contain headers and odd indices contain the corresponding section data.
        sort_key_function (function): A function that takes a header (string) as input and returns a numerical value for sorting.

    Returns:
        list: A list of headers and sections sorted by the headers, based on the sort_key_function.
    """

    # Pair headers with their correIsponding sections

    paired_sections = [(sections[i], sections[i + 1]) for i in range(0, len(sections), 2)]
    # Sort the paired sections by the numeric prefix in the header
    sorted_sections = sorted(paired_sections, key=lambda x: sort_key_fucntion(x[0]))

    # Flatten the list of sorted sections back into a single list
    sorted_sections_flat = [item for pair in sorted_sections for item in pair]
    logger.debug(f"sorted_sections: sorted sections: =\n" + "\n".join(k for k in sorted_sections_flat[::2]))

    return sorted_sections_flat


def column_name_formatter(header,label="H2O"):
    """
    Formats the column name by extracting a specified percentage from the header.

    Args:
        header (str): The header string from which to extract information.
        label (str): The label that identifies the component (e.g., "H2O", "D2O").

    Returns:
        str: The formatted column name.
    """

    header_components = header.split()
    # Extract date and H2O percentage from header
    date = header_components[0]
    logger.debug(f"date = {date}") 
    
    # Extract the specified label percentage using a regex pattern
    match = re.search(r'(\d+)%' + re.escape(label), header)
    if match:
        label_percentage = match.group(1).zfill(2)
        logger.debug(f"{label}_percentage= {label_percentage}")
    else:
        logger.error(f"Label '{label}' not found in header '{header}'")
        raise ValueError(f"Label '{label}' not found in header.")

    # Format the column name
    column_name = f"{label_percentage}{label}_{date}"
    logger.debug(f"column_name= {column_name}")
    
    return column_name

# Function to extract numeric prefix from the keys
def extract_numeric_prefix(key):
    numeric_prefix = ''.join(filter(str.isdigit, key.split('_')[0]))
    return int(numeric_prefix)


def file_to_DF(file_path):
    logger.info("converting file to DF")
    sections = section_split(file_path)
    sorted_sections = sort_sections(sections,extract_h2o_percentage )
    # Dictionary to store the extracted data
    data_dict = {}
    # Process each section
    for idx, header in enumerate(sorted_sections):
        if idx % 2 == 0:  # Ensure we're processing headers

            data_section = sorted_sections[idx + 1].strip()
            
            logger.info(f"Extracting data from section: {header}")
            data_extracted = process_section(data_section)
            data_x = data_extracted[0]
            data_y = data_extracted[1]

            column_name = column_name_formatter(header)
    
        #data_section = sections[i + 1].strip()
        #logger.debug(f"data_section = {data_section}")

        #
    #    # Process the section and get the data
        #data_x, data_y = process_section(data_section, header)
            logger.debug("#"*100)
            data_dict[column_name + "_x"] = data_x
            data_dict[column_name] = data_y
    
    ## Create a DataFrame from the dictionary
    logger.debug(f"data dict = \n" + "\n".join("{}\t{}".format(k, v) for k, v in data_dict.items()))
    return pd.DataFrame(data_dict)

def extract_t1_from_nova(file_path):

    logger.info("Extract_T1_from_mnova file")
    sections = section_split(file_path)
    data_dict = {}
    # Process each section
    for i in range(0, len(sections), 2):
    
        header, column_name = column_name_formatter(sections, i )
    
        data_section = sections[i + 1].strip()
        logger.debug(f"data_section = {data_section}")


    #    # Process the section and get the data
        G_values = re.findall(r'G\s*=\s*([\d.e-]+)', data_section)
        G_value = G_values[1] if len(G_values) >=2 else G_values[0]
        logger.debug(f"G_value: {G_value}")
        logger.debug("#"*100)
        data_dict[column_name] = float(G_value)
        indexs = [int(re.search(r'(\d+)H2O',key ).group(1)) for key in  data_dict.keys()]
        df = pd.DataFrame(data_dict.values(), index=indexs, columns=["G_value"]).sort_index()
        df["T1_mnova"] = 1/df["G_value"]
    return df

if __name__ == "__main__":
    
    FILE_NAME = "processed_data_5.txt"

#    G1_mnova=  extract_t1_from_nova(FILE_NAME)
    df = file_to_DF(FILE_NAME)
#
#    logger.warning(f" df columns = {df.columns}")
#
#    logger.info(f"T1 Values from MNOVA:\n{G1_mnova}")
#    logger.info(f"df:\n{df}")
#    logger.info(f"df columns :\n{df.columns}")
#    ## Graphs for T_1 from MNOVA
#    fig = plt.figure(1, figsize=(12,6))
#    plt.plot(G1_mnova["T1_mnova"], "*--")
#    plt.grid()
#    plt.xlabel(r"$H_2O$ %")
#    plt.ylabel(r"$T_1$ [s]")
#    plt.title(r"MNOVA $T_1$ values", fontsize="xx-large")
#
#
#
    ## Graphs for integral vs H2O
    # Filter by columns that dont end with "_x" 
    ndf = df.filter(regex="^(?!.*_x$).*", axis=1)
    logger.info(ndf)

    last_values = ndf.iloc[-1]
    logger.info(f"last_values {last_values}")
    
    percentage = [int(re.search(r'(\d+)H2O', ndf.columns[i]).group(1).zfill(3)) for i in range(len(ndf.columns))]
    logger.info("percentage: \n"+"\n".join(str(perc) for perc in percentage))
    percentage.sort()

    # Display the sorted Series
    fig = plt.figure(2, figsize=(12,6))
    plt.plot(percentage, last_values ,"*--")
    plt.grid()
    plt.xlabel(r"$H_2O$ %")
    plt.ylabel("Integral")
    plt.xticks(percentage)
    plt.title(r"Integral vs $H_2O$%", fontsize="xx-large")
    plt.show()

