import nbformat as nbf

# Read code
with open('1_data_preprocessing.py', 'r', encoding='utf-8') as f:
    prep_code = f.read()

with open('2_model_training.py', 'r', encoding='utf-8') as f:
    train_code = f.read()

with open('3_model_deployment.py', 'r', encoding='utf-8') as f:
    deploy_code = f.read()

# Build notebook
nb = nbf.v4.new_notebook()

# Add cells
nb.cells.append(nbf.v4.new_markdown_cell("# Wind Energy AI Data Processing & Training\nThis notebook demonstrates the end-to-end Machine Learning pipeline."))

nb.cells.append(nbf.v4.new_markdown_cell("## Step 1: Data Preprocessing\nThis cell handles cleaning the data, feature engineering (creating the Target **Energy** variable), and scaling the features."))
nb.cells.append(nbf.v4.new_code_cell(prep_code))

nb.cells.append(nbf.v4.new_markdown_cell("## Step 2: Model Training\nThis cell trains multiple models (Linear Regression, Random Forest, XGBoost) and prints out the evaluation metrics (R² and RMSE) so you can directly compare them."))
nb.cells.append(nbf.v4.new_code_cell(train_code))

nb.cells.append(nbf.v4.new_markdown_cell("## Step 3: Model Deployment & Thresholds\nThis cell defines the suitability logic, pulling down the best model and setting the mathematical thresholds based on percentiles."))
nb.cells.append(nbf.v4.new_code_cell(deploy_code))

# Write to file
with open('Wind_Energy_Model_Experiment.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
