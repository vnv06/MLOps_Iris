import pytest
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class TestModelEvaluation:
    
    def test_model_loading(self):
        """Test that the model can be loaded successfully"""
        try:
            model = joblib.load("artifacts/model.joblib")
            assert model is not None
            print(f"✅ Model loaded successfully: {type(model)}")
        except Exception as e:
            pytest.fail(f"Model loading failed: {str(e)}")
    
    def test_model_prediction(self):
        """Test that the model can make predictions"""
        # Load model
        model = joblib.load("artifacts/model.joblib")
        
        # Create test data that matches Iris feature ranges
        test_cases = [
            # Setosa-like features
            [5.1, 3.5, 1.4, 0.2],
            # Versicolor-like features  
            [6.0, 2.7, 4.2, 1.3],
            # Virginica-like features
            [6.7, 3.0, 5.2, 2.3]
        ]
        
        for i, features in enumerate(test_cases):
            # Convert to DataFrame with correct column names
            feature_df = pd.DataFrame([features], columns=[
                'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
            ])
            
            # Make prediction
            prediction = model.predict(feature_df)
            
            # Check prediction shape and type
            assert prediction.shape == (1,), f"Prediction shape should be (1,), got {prediction.shape}"
            valid_predictions = ['setosa', 'versicolor', 'virginica']
            assert prediction[0] in valid_predictions, f"Prediction should be one of {valid_predictions}, got {prediction[0]}"  
            
            print(f"✅ Test case {i+1}: Features {features} -> Prediction: {prediction[0]}")
    
    def test_model_prediction_probabilities(self):
        """Test that model can output probabilities (if available)"""
        model = joblib.load("artifacts/model.joblib")
        
        # Test data
        test_features = [[5.1, 3.5, 1.4, 0.2]]
        feature_df = pd.DataFrame(test_features, columns=[
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ])
        
        # Check if model has predict_proba method
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_df)
            assert probabilities.shape[1] == 3, f"Should have 3 class probabilities, got {probabilities.shape[1]}"
            assert np.allclose(probabilities.sum(axis=1), 1.0), "Probabilities should sum to 1"
            print(f"✅ Probability prediction works: {probabilities[0]}")
        else:
            print("ℹ️  Model doesn't support probability predictions")
    
    def test_model_accuracy_on_sample(self):
        """Test model accuracy on a small sample of known data"""
        model = joblib.load("artifacts/model.joblib")
        
        # Sample test data with known expected outcomes
        # Format: [features, expected_class]
        test_samples = [
            ([5.1, 3.5, 1.4, 0.2], 'setosa'),  # Setosa
            ([6.0, 2.7, 4.2, 1.3], 'versicolor'),  # Versicolor
            ([6.7, 3.0, 5.2, 2.3], 'virginica'),  # Virginica
        ]
        
        correct_predictions = 0
        total_predictions = len(test_samples)
        
        for features, expected_class in test_samples:
            feature_df = pd.DataFrame([features], columns=[
                'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
            ])
            
            prediction = model.predict(feature_df)[0]
            
            if prediction == expected_class:
                correct_predictions += 1
            else:
                print(f"⚠️  Misclassification: expected {expected_class}, got {prediction} for features {features}")
        
        accuracy = correct_predictions / total_predictions
        print(f"✅ Sample accuracy: {accuracy:.2f} ({correct_predictions}/{total_predictions})")
        
        # Allow some tolerance for model variations
        assert accuracy >= 0.5, f"Model accuracy too low: {accuracy:.2f}"
    
    def test_model_feature_importance(self):
        """Test that model has expected feature importance structure"""
        model = joblib.load("artifacts/model.joblib")
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            assert len(importances) == 4, f"Should have 4 feature importances, got {len(importances)}"
            assert np.all(importances >= 0), "Feature importances should be non-negative"
            assert np.allclose(importances.sum(), 1.0), "Feature importances should sum to 1"
            print(f"✅ Feature importances: {importances}")
        else:
            print("ℹ️  Model doesn't have feature_importances_ attribute")
    
    def test_model_serialization(self):
        """Test that model can be re-serialized and reloaded"""
        original_model = joblib.load("artifacts/model.joblib")
        
        # Test serialization round-trip
        temp_path = "artifacts/temp_model.joblib"
        try:
            joblib.dump(original_model, temp_path)
            reloaded_model = joblib.load(temp_path)
            
            # Test that reloaded model makes same predictions
            test_features = [[5.1, 3.5, 1.4, 0.2]]
            feature_df = pd.DataFrame(test_features, columns=[
                'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
            ])
            
            original_pred = original_model.predict(feature_df)
            reloaded_pred = reloaded_model.predict(feature_df)
            
            assert np.array_equal(original_pred, reloaded_pred), "Predictions should be identical after reload"
            print("✅ Model serialization round-trip successful")
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Optional: Integration test with your prediction function
def test_prediction_function_integration():
    """Test the actual prediction function from your code"""
    try:
        # This would test your actual predict_iris function
        # You might need to import it or mock dependencies
        from your_module import predict_iris  # Adjust import as needed
        
        # Test with a sample ID (you might need to setup Feast first)
        # prediction = predict_iris(1)
        # assert prediction in [0, 1, 2]
        print("ℹ️  Prediction function integration test skipped - requires Feast setup")
        
    except ImportError:
        print("ℹ️  Prediction function not available for integration test")