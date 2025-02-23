import re
import pandas as pd
from typing import List, Dict, Tuple
import os
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime
import time
from sklearn.preprocessing import MinMaxScaler
import textwrap

def parse_plp_file(plp_path: str) -> Tuple[Dict[int, str], List[Dict], Dict[str, str]]:
    """
    Parses a .plp file, extracting ClassDescription, SparseMatrix, and Index mappings.

    Args:
    - plp_path (str): Path to the .plp file.

    Returns:
    - class_description_dict: A dictionary of class descriptions.
    - sparse_matrix: A list of dictionaries representing each sparse matrix row.
    - index_mapping: A dictionary mapping index labels to locus names.
    """
    # Initialize dictionaries and lists to store mappings
    class_description_dict = {}
    sparse_matrix_data = []
    index_mapping = {}

    # Load the .plp file content
    with open(plp_path, 'r') as file:
        plp_content = file.readlines()

    # Stage 1: Parse ClassDescription section
    for line in plp_content:
        class_match = re.match(r"#ClassDescription\s+(\d+)\s+(\w+)", line)
        if class_match:
            key = int(class_match.group(1))
            description = class_match.group(2)
            class_description_dict[key] = description

    # Stage 2: Parse SparseMatrix section
    sparse_matrix_section = False
    for line in plp_content:
        if line.strip() == "###SparseMatrix":
            sparse_matrix_section = True
            continue
        if sparse_matrix_section and line.startswith('#'):
            continue  # Skip metadata lines within SparseMatrix

        if sparse_matrix_section and line.strip():
            tokens = line.strip().split()
            if tokens and tokens[0].isdigit():
                class_id = int(tokens[0])
                row_data = {'ClassDescription': class_description_dict.get(class_id, "Unknown")}
                valid_row = False  # Track if the row has valid data
                for token in tokens[1:]:
                    index_value = token.split(':')
                    if len(index_value) == 2:
                        index, value = index_value
                        try:
                            value_float = float(value)
                            row_data[index] = value_float
                            valid_row = True  # Mark the row as valid if we have valid data
                        except ValueError:
                            continue  # Skip non-numeric values
                if valid_row:
                    sparse_matrix_data.append(row_data)  # Only append if the row is valid

    # Stage 3: Parse Index section to extract locus names
    index_section = False
    for line in plp_content:
        if line.strip() == "###Index":
            index_section = True
            continue

        if line.strip() == "###SecondaryLabels" and index_section:
          break  # Stop processing further lines when ###SecondaryLabels is encountered


        if index_section and line.strip():
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                num_id = parts[0].strip()
                locus_full = parts[1]
                # Extract text between '|' characters
                locus_match = re.search(r'\|(.*?)\|', locus_full)
                locus = locus_match.group(1) if locus_match else locus_full
                index_mapping[num_id] = locus

    return class_description_dict, sparse_matrix_data, index_mapping

def plp_to_df(class_descriptions: Dict[int, str], sparse_matrix: List[Dict], index_mapping: Dict[str, str]):
    """
    Saves the parsed data to a CSV file.

    Args:
    - class_descriptions: A dictionary of class descriptions.
    - sparse_matrix: A list of sparse matrix data.
    - index_mapping: A dictionary mapping index labels to locus names.
    - output_path: Path where the CSV will be saved.
    """
    # Prepare data for DataFrame
    rows = []
    for row in sparse_matrix:
        if row['ClassDescription'] != "Unknown":  # Only include valid rows
            # Clean ClassDescription by removing underscores and specific prefixes
            clean_description = row['ClassDescription'].split('_')[0]
            translated_row = {'ClassDescription': clean_description}
            for index, value in row.items():
                if index != 'ClassDescription':
                    locus_name = index_mapping.get(index, index)  # Replace index with locus if available
                    # Only include in the DataFrame if locus name doesn't start with "contaminant_" or "Reverse_"
                    if not locus_name.startswith("contaminant_") and not locus_name.startswith("Reverse_"):
                        translated_row[locus_name] = value
            rows.append(translated_row)

    # Convert to DataFrame
    df = pd.DataFrame(rows).fillna(0)  # Fill NaN values with 0 or any other appropriate value
    df = df.rename(columns={'ClassDescription': 'Age/Protein'})
    return df


# Select the proteins with RFE
def selectProteins(train, test, nProteins, tol, epsilon, c, max_iter, selectedProteins, proteinsToIgnore):
    # Identify and filter the common columns between the datasets
    common_columns = train.columns.intersection(test.columns)
    common_columns = common_columns.drop('Age/Protein')  # Exclude the target column 'Age/Protein'

    print("all proteins:", len(common_columns))
    if (not proteinsToIgnore.empty):
      common_columns = common_columns.difference(proteinsToIgnore)
    if (not selectedProteins.empty):
      common_columns = common_columns[common_columns.isin(selectedProteins)]

    print("proteins considered:", len(common_columns))


    # Define features (X) and target (y) for the train dataset
    X_train = train[common_columns]
    y_train = train['Age/Protein']

    # Define features (X) and target (y) for the test dataset
    X_test = test[common_columns]

    # Perform Recursive Feature Elimination (RFE) with SVR to select the most important proteins
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))


    linear_model = SVR(kernel='linear', epsilon=epsilon, tol=tol, C=c, cache_size=20000, max_iter=max_iter)  # Set up the SVR model


    rfe = RFE(estimator=linear_model, n_features_to_select=nProteins)  # Initialize RFE to select 'nProteins' features
    rfe.fit(X_train_scaled, y_train)  # Fit RFE to the right dataset

    # Get the selected features and extract the corresponding protein data
    selected_features = rfe.support_  # Boolean mask indicating selected features
    selected_proteins = X_train.columns[selected_features]  # Column names of the selected proteins

    return selected_proteins  # Return the names of the selected proteins


def LeaveAgeOut(train, test1, test2, test3, selected_proteins, eps, tol, c, max_iter, verbose=False):

    # Identify and filter the common columns across datasets
    common_columns = train.columns.intersection(test1.columns).intersection(test2.columns).intersection(test3.columns)
    common_columns = common_columns.drop('Age/Protein')

    # Filter to get only the proteins that are common and selected
    common_and_selected_proteins = [protein for protein in selected_proteins if protein in common_columns]

    # Use only these common and selected proteins
    X_training = train[common_and_selected_proteins]
    y_training = train['Age/Protein']

    X_test1 = test1[common_and_selected_proteins]
    X_test2 = test2[common_and_selected_proteins]
    X_test3 = test3[common_and_selected_proteins]
    y_test_original1 = test1['Age/Protein']
    y_test_original2 = test2['Age/Protein']
    y_test_original3 = test3['Age/Protein']

    # Convert the data to NumPy arrays for processing
    X_training = X_training.to_numpy()
    X_test1 = X_test1.to_numpy()
    X_test2 = X_test2.to_numpy()
    X_test3 = X_test3.to_numpy()

    # Lists to store predicted and true values
    y_test_pred1 = []
    y_test_pred2 = []
    y_test_pred3 = []
    y_test_true1 = []
    y_test_true2 = []
    y_test_true3 = []


    unique_ages = pd.concat([y_training, y_test_original1, y_test_original2, y_test_original3]).unique()
    # Perform Leave-Age-Out cross-validation manually
    for i in unique_ages:
        # Separate sample i from the rest of the data
        ageToRemove = i

        if (verbose):
          print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")
          print("Age to remove:", ageToRemove)
          print("Current time:", datetime.now())
          #print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")


        # Remove instances with the specified age from training data
        indices_to_remove = np.where(y_training == ageToRemove)[0]
        X_train = np.delete(X_training, indices_to_remove, axis=0)
        y_train = np.delete(y_training, indices_to_remove, axis=0)

        # Get indices of the specified age in each test set
        first_index_equal_to_age1 = np.where(y_test_original1 == ageToRemove)[0]
        X_test1_filtered = X_test1[first_index_equal_to_age1]
        y_test1 = y_test_original1.values[first_index_equal_to_age1]

        first_index_equal_to_age2 = np.where(y_test_original2 == ageToRemove)[0]
        X_test2_filtered = X_test2[first_index_equal_to_age2]
        y_test2 = y_test_original2.values[first_index_equal_to_age2]

        first_index_equal_to_age3 = np.where(y_test_original3 == ageToRemove)[0]
        X_test3_filtered = X_test3[first_index_equal_to_age3]
        y_test3 = y_test_original3.values[first_index_equal_to_age3]

        # Skip iteration if there are no test samples with the specified age
        if y_test1.size == 0 and y_test2.size == 0 and y_test3.size == 0:
            continue


        if (verbose):
          print("X_train:", X_train)
          print("y_train:", y_train)
          print("X_test1_filtered:", X_test1_filtered)
          print("y_test1:", y_test1)
          print("X_test2_filtered:", X_test2_filtered)
          print("y_test2:", y_test2)
          print("X_test3_filtered:", X_test3_filtered)
          print("y_test3:", y_test3)
          print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")



        # Train the model
        ransac_model = SVR(kernel='linear', epsilon=eps, tol=tol, C=c, cache_size=20000, max_iter=max_iter)
        ransac_model.fit(X_train, y_train)


        # Predict the age for the removed individual and store the results
        if y_test1.size != 0:
            y_pred1 = ransac_model.predict(X_test1_filtered)
            y_test_pred1.append(y_pred1[0])
            y_test_true1.append(y_test1[0])
            if len(y_pred1) == 2:
                y_test_pred1.append(y_pred1[1])
                y_test_true1.append(y_test1[1])

        if y_test2.size != 0:
            y_pred2 = ransac_model.predict(X_test2_filtered)
            y_test_pred2.append(y_pred2[0])
            y_test_true2.append(y_test2[0])
            if len(y_pred2) == 2:
                y_test_pred2.append(y_pred2[1])
                y_test_true2.append(y_test2[1])

        if y_test3.size != 0:
            y_pred3 = ransac_model.predict(X_test3_filtered)
            y_test_pred3.append(y_pred3[0])
            y_test_true3.append(y_test3[0])
            if len(y_pred3) == 2:
                y_test_pred3.append(y_pred3[1])
                y_test_true3.append(y_test3[1])

    # Convert lists to Pandas Series for easier metric calculation
    y_test_pred1 = pd.Series(y_test_pred1)
    y_test_true1 = pd.Series(y_test_true1)
    y_test_pred2 = pd.Series(y_test_pred2)
    y_test_true2 = pd.Series(y_test_true2)
    y_test_pred3 = pd.Series(y_test_pred3)
    y_test_true3 = pd.Series(y_test_true3)

    # Calculate Mean Absolute Error (MAE)
    mae1 = mean_absolute_error(y_test_true1, y_test_pred1)
    mae2 = mean_absolute_error(y_test_true2, y_test_pred2)
    mae3 = mean_absolute_error(y_test_true3, y_test_pred3)

    # Calculate R² score
    r21 = r2_score(y_test_true1, y_test_pred1)
    r22 = r2_score(y_test_true2, y_test_pred2)
    r23 = r2_score(y_test_true3, y_test_pred3)

    # Create dataframes with original and predicted ages for each test set
    result_df1 = pd.DataFrame({
        'Idade Original': y_test_true1,
        'Idade Predita': y_test_pred1
    })
    result_df2 = pd.DataFrame({
        'Idade Original': y_test_true2,
        'Idade Predita': y_test_pred2
    })
    result_df3 = pd.DataFrame({
        'Idade Original': y_test_true3,
        'Idade Predita': y_test_pred3
    })

    return mae1, r21, result_df1, mae2, r22, result_df2, mae3, r23, result_df3




def Plot2(results1, results2, results3, mae1, r21, mae2, r22, mae3, r23):
    """
    Plots actual vs predicted ages for three datasets and prints evaluation metrics.
    """

    # Plot for Predict on Self
    plot_results2(results1['Idade Original'], results1['Idade Predita'], mae1, r21,
                 'Predict on Self: Actual vs Predicted Age')
    print('Predict on Self: Actual vs Predicted Age:')
    print(results1)

    # Plot for Predict on Bioactive
    plot_results2(results2['Idade Original'], results2['Idade Predita'], mae2, r22,
                 'Predict on Bioactive: Actual vs Predicted Age')
    print('Predict on Bioactive: Actual vs Predicted Age')
    print(results2)

    # Plot for Predict on Placebo
    plot_results2(results3['Idade Original'], results3['Idade Predita'], mae3, r23,
                 'Predict on Placebo: Actual vs Predicted Age')
    print('Predict on Placebo: Actual vs Predicted Age')
    print(results3)



def plot_results2(original, predicted, mae, r2, title, xlim=None, ylim=None):
    """
    Args:
    - original: Original/actual values.
    - predicted: Predicted values.
    - mae: Mean Absolute Error (MAE) value for display.
    - r2: R-squared value for display.
    - title: Title of the plot.
    - xlim: Optional x-axis limits.
    - ylim: Optional y-axis limits.
    """
    # Convert original and predicted to numeric to avoid type issues
    original = pd.to_numeric(original, errors='coerce')
    predicted = pd.to_numeric(predicted, errors='coerce')

    # Prepare data for sorting and processing
    data = pd.DataFrame({'Original': original, 'Predicted': predicted}).dropna()
    data = data.sort_values(by='Original')

    sorted_original = data['Original'].values
    sorted_predicted = data['Predicted'].values

    xlim = [20, 80]
    ylim = [20, 80]

    if xlim is None:
        xlim = (min(sorted_original), max(sorted_original))
    if ylim is None:
        ylim = (min(sorted_predicted), max(sorted_predicted))


    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(sorted_original, sorted_predicted, alpha=0.6, label='Data Points')
    plt.plot(xlim, xlim, color='r', linestyle='--', label='Ideal Age Prediction (y=x)')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xlabel('Chronological Age (years)')
    plt.ylabel('Predicted Proteomic Age (years)')
    plt.title(title)
    plt.legend()
    text_x = (xlim[0] + xlim[1]) / 2
    text_y = ylim[1] - (ylim[1] - ylim[0]) * 0.05
    plt.text(text_x, text_y, f'MAE: {mae:.2f}\nR²: {r2:.2f}', fontsize=12, color='black',
             ha='center', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    plt.show()






# Main execution -----------------------------------------------------------------------------------------------------------------------------------------------
# Set the plp directory path
directory = os.path.join(os.path.dirname(__file__), "..", "plp")

#initial selection of proteins. Keep empty pd.Index([]) to use all proteins
selectedProteins = pd.Index([])
proteinsToIgnore = pd.Index([])

# Display all rows
pd.set_option('display.max_rows', None)

# Process each .plp file and assign to distinct variables
people_class_descriptions, people_sparse_matrix, people_index_mapping = parse_plp_file(os.path.join(directory, "People.plp"))
placebo_class_descriptions, placebo_sparse_matrix, placebo_index_mapping = parse_plp_file(os.path.join(directory, "Placebo.plp"))
bioactive_class_descriptions, bioactive_sparse_matrix, bioactive_index_mapping = parse_plp_file(os.path.join(directory, "Bioactive.plp"))

# Convert to DataFrames
people = plp_to_df(people_class_descriptions, people_sparse_matrix, people_index_mapping)
placebo = plp_to_df(placebo_class_descriptions, placebo_sparse_matrix, placebo_index_mapping)
bioactive = plp_to_df(bioactive_class_descriptions, bioactive_sparse_matrix, bioactive_index_mapping)


print(people.columns)
print(placebo.columns)
print(bioactive.columns)


people['Age/Protein'] = people['Age/Protein'].astype(int)
placebo['Age/Protein'] = placebo['Age/Protein'].astype(int)
bioactive['Age/Protein'] = bioactive['Age/Protein'].astype(int)


all_columns = set(people.columns).union(placebo.columns).union(bioactive.columns)
people = people.reindex(columns=all_columns, fill_value=0)
placebo = placebo.reindex(columns=all_columns, fill_value=0)
bioactive = bioactive.reindex(columns=all_columns, fill_value=0)


numbersOfProteins = 120
tols = 0.1
epsilons = 0.001
cs = 10**10
max_iters=10000000

selectedProteins = selectProteins(train=people, test=people, selectedProteins=selectedProteins, proteinsToIgnore=proteinsToIgnore, nProteins=numbersOfProteins, tol=tols, epsilon=epsilons, c=cs, max_iter=max_iters)
mae1, r21, results1, mae2, r22, results2, mae3, r23, results3 = LeaveAgeOut(people, people, bioactive, placebo, selectedProteins, epsilons, tols, cs, max_iters, verbose=True)


print("=================================================================================================")
#PLOTTING RESULTS ======================================================================================

Plot2(results1,results2,results3, mae1, r21, mae2, r22, mae3, r23)
