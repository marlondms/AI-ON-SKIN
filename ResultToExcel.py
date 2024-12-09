import os
import pandas as pd

# Function to parse a specific file
def parse_file(file_path):
    """
    Reads a text file and extracts MAE and R² values for each group.
    """
    results = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(':')
            group_name = parts[0].strip()
            values = list(map(float, parts[1].split()))
            results[group_name] = {
                'MAE': values[0],
                'R2': values[1]
            }
    return results

# Function to extract the number of proteins from the file name
def extract_protein_count(file_name):
    """
    Extracts the number of proteins from the file name.
    Example: '100.txt' returns 100.
    """
    base_name = os.path.splitext(file_name)[0]
    if base_name.isdigit():
        return int(base_name)
    return None

# Function to process all files in a directory
def process_directory(directory_path):
    """
    Reads all files in the directory and organizes the extracted data.
    """
    all_data = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(directory_path, file_name)
            protein_count = extract_protein_count(file_name)
            if protein_count is not None:
                parsed_data = parse_file(file_path)
                row = {'Protein Count': protein_count}
                for group in ['PreTreatment', 'Bioester', 'Placebo']:
                    if group in parsed_data:
                        row[f'{group} MAE'] = parsed_data[group]['MAE']
                        row[f'{group} R2'] = parsed_data[group]['R2']
                    else:
                        row[f'{group} MAE'] = None
                        row[f'{group} R2'] = None
                all_data.append(row)
    return pd.DataFrame(all_data)

# Main execution block
if __name__ == "__main__":
    # Specify the directory containing .txt files
    directory_path = r'C:\Users\marlo\Desktop\Results'

    # Process the directory and generate an Excel file
    if os.path.isdir(directory_path):
        try:
            df = process_directory(directory_path)
            if not df.empty:
                # Sort the dataframe by Protein Count
                df = df.sort_values(by='Protein Count', ascending=True)
                output_path = os.path.join(directory_path, "results_summary.xlsx")
                df.to_excel(output_path, index=False)
                print(f"Excel file generated successfully: {output_path}")
            else:
                print("No valid data found in the specified directory.")
        except Exception as e:
            print(f"An error occurred while processing files: {e}")
    else:
        print("Invalid directory path. Please try again.")
