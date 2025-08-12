import datalake.utilities as util
      
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import os
import re

class Plotter:
    """
    This class creates plots
    """

    def __init__(self) -> None:
        pass

    def retrieve_scores_from_classification_report_list(self, dict_with_lists_of_classification_reports: dict) -> dict:
        """
        We output the results of the neural networks in a list of classification reports per meter/experiment.
        This function reads through all the meters/experiments and retrieves the relevant f1 scores for us to plot.
        """
        # Calculate average F1 scores for each category per meter
        average_f1_scores_per_meter = {}

        for meter, entries in dict_with_lists_of_classification_reports.items():
            average_f1_scores = {
                "long": sum(entry["long"]["f1-score"] for entry in entries) / len(entries),
                "short": sum(entry["short"]["f1-score"] for entry in entries) / len(entries),
                "elision": sum(entry["elision"]["f1-score"] for entry in entries) / len(entries),
            }
            average_f1_scores_per_meter[meter] = average_f1_scores

        return average_f1_scores_per_meter

    def find_latest_timestamped_json(self, directory: str) -> str:
        """
        Returns the latest timestamped json in the given directory
        """
        # List all entries in the directory
        entries = os.listdir(directory)

        # Define a regular expression pattern to match the timestamp format in filenames
        pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}\.json$')

        # Filter files that match the timestamp pattern and have a .json extension
        timestamped_files = [entry for entry in entries if pattern.fullmatch(entry)]

        if not timestamped_files:
            return None

        # Sort the timestamped files and return the latest one
        latest_file = sorted(timestamped_files, reverse=True)[0]

        return latest_file


if __name__ == "__main__":
    plotter = Plotter()

    # experiment = 'run_lstm_on_each_meter_type'
    experiment = 'run_combination_lstm_on_smaller_meters'

    latest_experiment_json = plotter.find_latest_timestamped_json(f'neural_networks/experiments/{experiment}')
    f1_scores_per_meter = plotter.retrieve_scores_from_classification_report_list(util.read_json(f"neural_networks/experiments/{experiment}/{latest_experiment_json}"))
    # Display the results
    for meter, scores in f1_scores_per_meter.items():
        print(f"{meter}: {scores}")

    # Convert the data to a DataFrame for the heatmap
    df_heatmap = pd.DataFrame(f1_scores_per_meter).T

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_heatmap, annot=True, cmap="Blues", cbar_kws={"label": "Average F1 Score"})
    plt.title("Average F1 Scores by Meter and Category")
    plt.ylabel("Meter")
    plt.xlabel("Category")
    plt.show()
