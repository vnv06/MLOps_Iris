import pytest
import pandas as pd
import numpy as np
import os

class TestDataValidation:
    
    def test_data_file_exists(self):
        """Test that the data file exists and can be loaded"""
        data_path = "data/iris.csv"
        assert os.path.exists(data_path), f"Data file not found at {data_path}"
        
        df = pd.read_csv(data_path)
        assert len(df) > 0, "Data file should not be empty"
        print(f"✅ Data file loaded successfully with {len(df)} rows")
    
    def test_feature_columns_exist(self):
        """Test that required feature columns exist"""
        df = pd.read_csv("data/iris.csv")
        
        expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for col in expected_columns:
            assert col in df.columns, f"Missing expected column: {col}"
        print("✅ All expected feature columns present")
    
    def test_feature_ranges(self):
        """Test that feature values are within expected biological ranges"""
        df = pd.read_csv("data/iris.csv")
        
        # Iris dataset typical ranges
        feature_ranges = {
            'sepal_length': (4.0, 8.0),
            'sepal_width': (2.0, 4.5), 
            'petal_length': (1.0, 7.0),
            'petal_width': (0.1, 2.5)
        }
        
        for feature, (min_val, max_val) in feature_ranges.items():
            assert df[feature].min() >= min_val, f"{feature} values below expected range"
            assert df[feature].max() <= max_val, f"{feature} values above expected range"
            print(f"✅ {feature} within range [{min_val}, {max_val}]")
    
    def test_no_null_values(self):
        """Test that there are no null values in the dataset"""
        df = pd.read_csv("data/iris.csv")
        
        null_count = df.isnull().sum().sum()
        assert null_count == 0, f"Found {null_count} null values in the dataset"
        print("✅ No null values found in dataset")
    
    def test_data_shape(self):
        """Test that dataset has expected shape"""
        df = pd.read_csv("data/iris.csv")
        
        # Iris dataset typically has 150 samples and 4-5 columns
        assert df.shape[0] >= 100, f"Too few samples: {df.shape[0]}"
        assert df.shape[1] >= 4, f"Too few columns: {df.shape[1]}"
        print(f"✅ Dataset shape: {df.shape}")
    
    def test_target_variable(self):
        """Test target variable if present"""
        df = pd.read_csv("data/iris.csv")
        
        # Check if target column exists (could be 'species', 'target', etc.)
        target_columns = ['species', 'target', 'class']
        target_col = None
        
        for col in target_columns:
            if col in df.columns:
                target_col = col
                break
        
        if target_col:
            unique_classes = df[target_col].nunique()
            assert unique_classes == 3, f"Expected 3 classes, got {unique_classes}"
            print(f"✅ Target variable '{target_col}' has {unique_classes} classes")
        else:
            print("ℹ️  No target variable found (this is OK for feature data)")