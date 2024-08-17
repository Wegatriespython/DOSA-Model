import pandas as pd
import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
import os
import matplotlib.pyplot as plt

def perform_sensitivity_analysis(df_path):
    # Load the DataFrame
    df = pd.read_csv(df_path)

    # Define the problem
    problem = {
        'num_vars': 4,
        'names': ['budget', 'current_labor', 'expected_price', 'wage'],
        'bounds': [
            [df['budget'].min(), df['budget'].max()],
            [df['current_labor'].min(), df['current_labor'].max()],
            [df['expected_price'].min(), df['expected_price'].max()],
            [df['wage'].min(), df['wage'].max()]
        ]
    }

    # Generate samples
    param_values = saltelli.sample(problem, 1024)

    # Run the model (in this case, we'll use the data we have)
    Y = np.zeros(param_values.shape[0])
    for i, X in enumerate(param_values):
        # Find the nearest neighbor in our dataset
        distances = np.sum((df[problem['names']] - X)**2, axis=1)
        nearest_index = np.argmin(distances)
        Y[i] = df.iloc[nearest_index]['optimal_production']

    # Perform analysis
    Si = sobol.analyze(problem, Y)

    # Print results
    print("First-order sensitivity indices:")
    for name, S1 in zip(problem['names'], Si['S1']):
        print(f"{name}: {S1:.4f}")

    print("\nTotal-order sensitivity indices:")
    for name, ST in zip(problem['names'], Si['ST']):
        print(f"{name}: {ST:.4f}")

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # First-order indices
    ax1.bar(problem['names'], Si['S1'])
    ax1.set_title('First-order Sensitivity Indices')
    ax1.set_ylabel('Sensitivity')

    # Total-order indices
    ax2.bar(problem['names'], Si['ST'])
    ax2.set_title('Total-order Sensitivity Indices')
    ax2.set_ylabel('Sensitivity')

    plt.tight_layout()
    plt.savefig('sensitivity_indices.png')
    plt.close()

    # Calculate elasticities
    elasticities = {}
    for var in problem['names']:
        # Calculate percentage changes
        pct_change_x = df[var].pct_change()
        pct_change_y = df['optimal_production'].pct_change()

        # Calculate elasticity
        elasticity = pct_change_y / pct_change_x

        # Remove infinite and NaN values
        elasticity = elasticity[np.isfinite(elasticity)]

        # Store mean elasticity
        elasticities[var] = elasticity.mean()

    print("\nMean Elasticities:")
    for var, elast in elasticities.items():
        print(f"{var}: {elast:.4f}")

if __name__ == "__main__":
    file_path = "V:\Python Port\Src\profit_maximization_sensitivity.csv"
    perform_sensitivity_analysis(file_path)  # Replace with your actual file path
