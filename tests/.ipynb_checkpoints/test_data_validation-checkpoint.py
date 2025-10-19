import pytest
import pandas as pd
import numpy as np
from great_expectations.dataset import PandasDataset
import os

class TestDataValidation:
    
    def test_feature_ranges(self):
        """Test that feature values are within expected ranges"""
        # Load actual data from DVC-tracked data folder
        data_path = "data/iris.csv"  # Adjust path based on your actual data structure
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Fallback for testing structure when data isn't available
            pytest.skip("Data file not available - skipping range validation")
            return
        
        dataset = PandasDataset(df)
        
        # Test value ranges based on Iris dataset characteristics
        assert dataset.expect_column_values_to_be_between(
            'sepal_length', 4.0, 8.0
        ).success
        assert dataset.expect_column_values_to_be_between(
            'sepal_width', 2.0, 4.5
        ).success
        assert dataset.expect_column_values_to_be_between(
            'petal_length', 1.0, 7.0
        ).success
        assert dataset.expect_column_values_to_be_between(
            'petal_width', 0.1, 2.5
        ).success
    
    def test_no_null_values(self):
        """Test that there are no null values in features"""
        data_path = "data/iris.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            pytest.skip("Data file not available - skipping null validation")
            return
        
        dataset = PandasDataset(df)
        
        # Test for null values in feature columns
        feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for column in feature_columns:
            if column in df.columns:
                assert dataset.expect_column_values_to_not_be_null(column).success
    
    def test_target_distribution(self):
        """Test that target variable has expected classes"""
        data_path = "data/iris.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            pytest.skip("Data file not available - skipping target validation")
            return
        
        if 'species' in df.columns or 'target' in df.columns:
            target_col = 'species' if 'species' in df.columns else 'target'
            dataset = PandasDataset(df)
            
            # Test that target has expected number of classes (3 for Iris)
            unique_classes = df[target_col].nunique()
            assert unique_classes == 3, f"Expected 3 classes, got {unique_classes}"
    
    def test_data_shape(self):
        """Test that dataset has expected shape"""
        data_path = "data/iris.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            pytest.skip("Data file not available - skipping shape validation")
            return
        
        # Iris dataset typically has 150 samples
        assert df.shape[0] > 0, "Dataset should have at least one sample"
        assert df.shape[1] >= 4, "Dataset should have at least 4 feature columns"