import os
import csv
from pathlib import Path

def get_working_path(dev_mode = False):
    if dev_mode:
        # development modes
        # get the parent directory
        p = Path(__file__).parents[2]

        # p to string
        p = str(p)

        result = p
    else:
        # production/NetLogo mode
        # get the parent directory
        result = os.getcwd()

    return result

def write_to_csv(filename, header, data, start_date_string, folder_name="result"):
    folder_path = os.path.join(folder_name, start_date_string)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    filename = os.path.join(folder_path, filename)
    file_exists = os.path.exists(filename)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)

        writer.writerow(data)

if __name__ == "__main__":
    print("Current Directory:", CURRENT_DIRECTORY)
    print("Parent Directory:", PARENT_DIRECTORY)
