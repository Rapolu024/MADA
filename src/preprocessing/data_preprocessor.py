
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Configure logging
logger = logging.getLogger(__name__)

class MedicalDataPreprocessor:
    def __init__(self, numerical_features, categorical_features, test_size=0.2, random_state=42):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self):
        # Create preprocessing pipelines for both numerical and categorical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Bundle preprocessing for numerical and categorical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor

    def preprocess(self, data):
        # Separate features and target variable
        X = data.drop('target', axis=1)
        y = data['target']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Apply preprocessing
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        logger.info(f"Data preprocessed successfully. Shape of X_train: {X_train_processed.shape}")

        return X_train_processed, X_test_processed, y_train, y_test, self.preprocessor

# Example usage:
if __name__ == '__main__':
    # Sample DataFrame
    data = {
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'M'],
        'symptom1': [1, 0, 1, 0, 1, 0, 1, 1],
        'symptom2': [0, 1, 0, 1, 0, 1, 0, 1],
        'target': [0, 1, 0, 1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    numerical_features = ['age', 'symptom1', 'symptom2']
    categorical_features = ['gender']

    preprocessor = MedicalDataPreprocessor(numerical_features, categorical_features)
    try:
        X_train_p, X_test_p, y_train, y_test, proc = preprocessor.preprocess(df)
        logger.info("Preprocessing complete.")
        logger.info(f"X_train_processed shape: {X_train_p.shape}")
        logger.info(f"X_test_processed shape: {X_test_p.shape}")
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")

