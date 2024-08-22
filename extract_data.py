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


# Function to process each section of the data
def process_section(data_section, header):
    logger.info(f"Extracting data from section: {header}") 
    lines = data_section.strip().split('\n')
    logger.debug(f"process_section:lines = {lines}")

    data_x = []
    data_y = []
    for line in lines[2:12]:  # Extract only the 10 data points
        columns = line.split()
        logger.debug(f"process_section: columns = {columns}")
        data_x.append(float(columns[1]))  # Extract the third column (X(X))
        data_y.append(float(columns[2]))  # Extract the third column (Y(X))

    logger.debug(f"process_section: data_y = {data_y}\n data_x[-1] = {data_x[-1]}")
    return data_x, data_y

def section_split(file_path):
    # Read the file content
    with open(file_path, 'r') as file:
        logger.info(f"reading file {file_path}")
        content = file.read()
        logger.debug(f"content\n {content}")

    # Split the content by header lines
    sections = re.split(r'(\d{6}-\d{6} T1 \(\d+%H2O\d+%D2O\(non degassed\)\)).*', content)[1:]
    logger.debug(f"sections = {sections[:4]}")
    logger.info(f"Number of sections : {len(sections)//2}") 

    return sections

def column_name_formatter(sections, i ):
    header = sections[i].strip()
    logger.debug(f"header = {header}")

    # Extract date and H2O percentage from header
    date = header.split()[0]
    logger.debug(f"date = {date}") 

    h2o_percentage = re.search(r'(\d+)%H2O', header).group(1).zfill(2)
    logger.debug(f"h2o_percentage= {h2o_percentage}") 

    column_name = f"{h2o_percentage}H2O_{date}"
    logger.debug(f"column_name= {column_name}")
    
    return header, column_name


def file_to_DF(file_path):
    sections = section_split(file_path)
    # Dictionary to store the extracted data
    data_dict = {}
    # Process each section
    for i in range(0, len(sections), 2):
    
        header, column_name = column_name_formatter(sections, i )
    
        data_section = sections[i + 1].strip()
        logger.debug(f"data_section = {data_section}")

        #
    #    # Process the section and get the data
        data_x, data_y = process_section(data_section, header)
        logger.debug("#"*100)
        data_dict[column_name + "_x"] = data_x
        data_dict[column_name] = data_y


    ## Create a DataFrame from the dictionary
    logger.debug(f"data dict = {data_dict}")
    return pd.DataFrame(data_dict, columns=sorted(set(data_dict.keys())))

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
        G_value = re.findall(r'G\s*=\s*([\d.e-]+)', data_section)[1]
        logger.debug(f"G_value: {G_value}")
        logger.debug("#"*100)
        data_dict[int(column_name[0:2])] = float(G_value)
        df = pd.DataFrame(data_dict.values(), index=data_dict.keys(), columns=["G_value"]).sort_index()
        df["T1_mnova"] = 1/df["G_value"]
    logger.error(data_dict)
    return df
   

if __name__ == "__main__":
    
    G1_mnova=  extract_t1_from_nova("processed_data.txt")
    ## Graphs for T_1 from MNOVA
    fig = plt.figure(1, figsize=(12,6))
    plt.plot(G1_mnova["T1_mnova"], "*--")
    plt.grid()
    plt.xlabel(r"$H_2O$ %")
    plt.ylabel(r"$T_1$ [s]")
    plt.title(r"MNOVA $T_1$ values", fontsize="xx-large")

    df = file_to_DF("processed_data_2.txt")
    ndf = df.filter(regex="^(?!.*_x$).*", axis=1)
    logger.info(ndf)

    last_values = ndf.iloc[-1]
    logger.info(f"last_values {last_values}")
    
    percentage = [int(re.search(r'(\d+)H2O', ndf.columns[i]).group(1).zfill(2)) for i in range(len(ndf.columns))]
    percentage.sort()

    # Display the sorted Series
    fig = plt.figure(2, figsize=(12,6))
    plt.plot(percentage, last_values.sort_index(),"*--")
    plt.grid()
    plt.xlabel(r"$H_2O$ %")
    plt.ylabel("Integral")
    plt.title(r"Integral vs $H_2O$%", fontsize="xx-large")
    plt.show()
