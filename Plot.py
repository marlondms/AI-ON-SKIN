import os
import matplotlib.pyplot as plt

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
    all_data = {}
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):  # Process only .txt files
            file_path = os.path.join(directory_path, file_name)
            protein_count = extract_protein_count(file_name)
            if protein_count is not None:
                all_data[f"{protein_count} proteins"] = parse_file(file_path)
    return dict(sorted(all_data.items(), key=lambda x: int(x[0].split()[0])))  # Sort by protein count

# Function to find the optimal point
def find_optimal_point(all_data):
    """
    Finds the data point with the maximum R² and minimum MAE across all groups.
    """
    optimal_point = {
        "group": None,
        "protein_count": None,
        "MAE": float('inf'),
        "R2": float('-inf')
    }
    
    for protein_label, data in all_data.items():
        protein_count = int(protein_label.split()[0])
        for group, metrics in data.items():
            mae = metrics["MAE"]
            r2 = metrics["R2"]
            
            # Update if this point has a better combination of higher R² and lower MAE
            if r2 > optimal_point["R2"] or (r2 == optimal_point["R2"] and mae < optimal_point["MAE"]):
                optimal_point = {
                    "group": group,
                    "protein_count": protein_count,
                    "MAE": mae,
                    "R2": r2
                }
    return optimal_point


# Function to create a plot with consistent colors for MAE (circles) and R² (lines)
def plot_dual_axis_with_visible_x_numbers(all_data):
    """
    Creates a plot with two Y-axes (MAE and R²) and X-axis labels based on the number of proteins.
    Ensures consistent colors for circles (MAE) and lines (R²).
    """
    groups = ["PreTreatment", "Bioester", "Placebo"]
    group_colors = {"PreTreatment": "blue", "Bioester": "orange", "Placebo": "green"}  # Define consistent colors
    protein_counts = [int(label.split()[0]) for label in all_data.keys()]  # Extract integer numbers from the labels
    protein_labels = [str(count) for count in protein_counts]  # Convert to strings for labeling
    
    # Initialize MAE and R² values for each group
    mae_values = {group: [] for group in groups}
    r2_values = {group: [] for group in groups}

    # Organize values for plotting
    for protein_label, data in all_data.items():
        for group in groups:
            mae_values[group].append(data[group]["MAE"])
            r2_values[group].append(data[group]["R2"])
    
    # Create the plot with two Y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Left Y-axis for MAE
    ax1.set_xlabel("Number of Proteins")
    ax1.set_ylabel("MAE", color='blue')
    for group, values in mae_values.items():
        ax1.plot(
            protein_counts, 
            values, 
            marker='o', 
            linestyle='None', 
            label=f"{group} (MAE)", 
            color=group_colors[group]
        )
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), title="MAE Groups")
    ax1.grid(True)
    
    # Adjust X-axis ticks dynamically
    tick_step = max(1, len(protein_counts) // 100)  # Show approximately 10 ticks
    ax1.set_xticks(protein_counts[::tick_step])  # Set ticks at regular intervals
    ax1.set_xticklabels(protein_labels[::tick_step], rotation=45, ha='right')  # Rotate for better visibility

    # Right Y-axis for R²
    ax2 = ax1.twinx()
    ax2.set_ylabel("R²", color='red')
    for group, values in r2_values.items():
        ax2.plot(
            protein_counts, 
            values, 
            linestyle='-', 
            label=f"{group} (R²)", 
            color=group_colors[group]
        )
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), title="R² Groups")

    # Plot title
    plt.title("Comparison of MAE and R² by Number of Proteins")
    plt.tight_layout()  # Adjust layout to prevent label overlap
    plt.show()

# Main execution block
if __name__ == "__main__":
    # Specify the directory containing .txt files
    directory_path = r'C:\Users\marlo\Desktop\Results'

    # Process the directory and generate the plot
    if os.path.isdir(directory_path):
        try:
            all_data = process_directory(directory_path)
            if all_data:
                optimal_point = find_optimal_point(all_data)
                print(f"Optimal Point: Group = {optimal_point['group']}, "
                      f"Protein Count = {optimal_point['protein_count']}, "
                      f"MAE = {optimal_point['MAE']}, R² = {optimal_point['R2']}")
                plot_dual_axis_with_visible_x_numbers(all_data)
            else:
                print("No valid data found in the specified directory.")
        except Exception as e:
            print(f"An error occurred while processing files: {e}")
    else:
        print("Invalid directory path. Please try again.")
