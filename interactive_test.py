#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
#import extract_data as xd
#from scipy.optimize import curve_fit
import pandas as pd
import re
import ethanol_graph as eg 
plt.style.use('./graphs_style.mplstyle')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

def main_plots(fitting_df : pd.DataFrame , arguments: list):
    """
    Plots the main T1 and Integral plots. Includes filtering and fit inclusion loggic
    Args:
        fitting_df: DataFrame with the data to plot 
        arguments: list with the arguments required
    Returns:
        fig.
    """


    fitting = False
    change_axis = False

    if "f" in arguments:
        fitting = True
        change_axis = True 

    if  "filter" in arguments:
        try:
            # Find the index of "filter" in the arguments list
            filter_index = arguments.index("filter")

            # Ensure there is an argument following "filter" for the range
            if filter_index + 1 < len(arguments):
                filter_arg = arguments[filter_index + 1]  # Get the filter range (e.g., "10-20")

                # Match the range format 'min-max' using regex
                match = re.match(r'(\d+)-(\d+)', filter_arg)

                if match:
                    min_val = match.group(1)
                    max_val = match.group(2)

                    # Filter the DataFrame by the range
                    fitting_df = fitting_df.loc[min_val:max_val]
                    logger.debug(f"Filtered DataFrame for range {min_val}-{max_val}")
            else:
                logger.error("Invalid filter range format. Expected 'min-max'.")
        except IndexError:
            logger.error("Filter specified but no filter value provided.")
        except KeyError:
            logger.error(f"Filter key {arguments[1]} not found in DataFrame.")
   
    if "integral" in arguments:
        eg.integral_plot(fitting_df, change_xaxis = change_axis, fitting = fitting)
    else :
        eg.integral_plot(fitting_df, change_xaxis = change_axis, fitting = fitting)
        eg.T1_plot(fitting_df, change_xaxis = change_axis, fitting = fitting)



def main_loop():
        
    file  = input("File number to extract data :")
    # Enable interactive mode for non-blocking plots
    plt.ion()  # Turn on interactive mode

    # Extracting data from file
    df = eg.ethanol_file_to_df("ethanol_data_"+file+".txt")
    
    # Calculating fitting to T1 Curve from data.
    fitting_df = eg.ethanol_t1_calc(df)
    
    # Filtering df to obtain T1 Values and Total magnetization from fitting calculations.
    magentization_df = fitting_df.filter(regex="^sample$|_a.*", axis=1)
    logger.info(f"data frame filtered with magentization:\n {magentization_df}")
    
    T1_df = fitting_df.filter(regex="^(?!.*_a).*", axis=1)
    logger.info(f"data frame filtered with T1 values :\n {T1_df}")
    
    # Run initial plots
    eg.integral_plot(fitting_df)
    eg.T1_plot(fitting_df)
    
    plt.pause(0.001)  # Use pause to display the plots in interactive mode
    
    while True:
        print("\nEnter a command: ")
        print("1: Run plt_fitting_2 with specific sample")
        print("2: Show integral_plot and T1_plot")
        print("3: Print T1 and M values")
        print("4: Exit")

        # Get user input
        user_input_raw = input("Your choice: ").strip()
        
        # Split the input into the command and any additional arguments
        split_input = user_input_raw.split()
        user_input = split_input[0]  # The first part is the command (e.g., "1", "2")
        arguments = split_input[1:]  # The remaining part (if any) are arguments
       
        if user_input == "1":
            sample_id = input("Enter the sample ID (e.g., '02'): ").strip()
            try:
                # Call the plt_fitting_2 with the provided sample ID
                eg.plt_fitting_2([sample_id], fitting_df, df)
                plt.pause(0.001)  # Non-blocking pause to update the plot
            except Exception as e:
                print(f"Error during plotting: {e}")
        
        elif user_input == "2":
            # Re-run the integral and T1 plots
            main_plots(fitting_df, arguments)
            plt.pause(0.001)  # Update the plot in interactive mode
        
        elif user_input == "3":
            logger.info(f"data frame filtered with magentization:\n {magentization_df}")
            logger.info(f"data frame filtered with T1 values :\n {T1_df}")
        elif user_input == "4" or user_input.lower() == "exit":
            print("Exiting the program.")
            break
        
        else:
            print("Invalid input, please try again.")
    
    plt.ioff()  # Turn off interactive mode when exiting

if __name__ == "__main__":
    

    main_loop()


