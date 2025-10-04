import pandas as pd
import numpy as np

def augment_iris_data():
    # Read original data
    original_data = pd.read_csv('data/iris.csv')
    
    print(f"Original data shape: {original_data.shape}")
    
    # Create a copy for augmentation
    augmented_data = original_data.copy()
    
    # Add calculated features as a fifth column
    # Feature 1: Sepal area to petal area ratio
    augmented_data['sepal_petal_area_ratio'] = (
        augmented_data['sepal_length'] * augmented_data['sepal_width']
    ) / (
        augmented_data['petal_length'] * augmented_data['petal_width'] + 1e-8  # Avoid division by zero
    )
    
    # Feature 2: Petal to sepal length ratio  
    augmented_data['petal_sepal_length_ratio'] = (
        augmented_data['petal_length'] / (augmented_data['sepal_length'] + 1e-8)
    )
    
    # Feature 3: Overall size metric (Euclidean-like combination)
    augmented_data['overall_size'] = np.sqrt(
        augmented_data['sepal_length']**2 + 
        augmented_data['sepal_width']**2 + 
        augmented_data['petal_length']**2 + 
        augmented_data['petal_width']**2
    )
    
    # Feature 4: Sepal shape factor (aspect ratio)
    augmented_data['sepal_shape_factor'] = (
        augmented_data['sepal_length'] / (augmented_data['sepal_width'] + 1e-8)
    )
    
    # Feature 5: Petal shape factor (aspect ratio)
    augmented_data['petal_shape_factor'] = (
        augmented_data['petal_length'] / (augmented_data['petal_width'] + 1e-8)
    )
    
    # Save augmented data
    augmented_data.to_csv('data/iris_augmented.csv', index=False)
    
    print(f"Augmented data shape: {augmented_data.shape}")
    print("New features added:")
    print("- sepal_petal_area_ratio")
    print("- petal_sepal_length_ratio") 
    print("- overall_size")
    print("- sepal_shape_factor")
    print("- petal_shape_factor")
    print("\nFirst 3 rows with new features:")
    print(augmented_data.head(3))
    
    # Save feature descriptions
    feature_descriptions = {
        'sepal_petal_area_ratio': 'Ratio of sepal area to petal area',
        'petal_sepal_length_ratio': 'Ratio of petal length to sepal length',
        'overall_size': 'Combined size metric using Euclidean distance',
        'sepal_shape_factor': 'Aspect ratio of sepal (length/width)',
        'petal_shape_factor': 'Aspect ratio of petal (length/width)'
    }
    
    with open('data/feature_descriptions.txt', 'w') as f:
        for feature, description in feature_descriptions.items():
            f.write(f"{feature}: {description}\n")

if __name__ == "__main__":
    augment_iris_data()