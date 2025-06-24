# stockout_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from lightgbm import LGBMClassifier
import joblib

# Step 1: Load data
df = pd.read_csv('Walmart.csv')

# Step 2: Feature Engineering
df['stock_to_sales_ratio'] = df['inventory_level'] / (df['quantity_sold'] + 1)
df['demand_surge'] = df['actual_demand'] - df['forecasted_demand']
df['rolling_avg_sales'] = df.groupby(['store_id', 'product_id'])['quantity_sold'].transform(lambda x: x.rolling(3, 1).mean())
df['sales_vs_reorder'] = df['quantity_sold'] - df['reorder_point']
df['demand_to_stock_ratio'] = df['actual_demand'] / (df['inventory_level'] + 1)

df.dropna(inplace=True)

# Step 3: Feature selection
features = [
    'inventory_level', 'reorder_point', 'reorder_quantity',
    'quantity_sold', 'forecasted_demand', 'actual_demand',
    'supplier_lead_time', 'promotion_applied', 'promotion_type',
    'holiday_indicator', 'weather_conditions', 'weekday', 'store_location',
    'stock_to_sales_ratio', 'demand_surge', 'rolling_avg_sales',
    'sales_vs_reorder', 'demand_to_stock_ratio'
]
target = 'stockout_indicator'

X = df[features]
y = df[target]

# Step 4: Upsample minority class
data = pd.concat([X, y], axis=1)
majority = data[data[target] == 0]
minority = data[data[target] == 1]

minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
balanced_data = pd.concat([majority, minority_upsampled])
X = balanced_data[features]
y = balanced_data[target]

# Step 5: Column types
categorical_features = ['promotion_type', 'weather_conditions', 'weekday', 'store_location']
numerical_features = [
    'inventory_level', 'reorder_point', 'reorder_quantity',
    'quantity_sold', 'forecasted_demand', 'actual_demand',
    'supplier_lead_time', 'stock_to_sales_ratio', 'demand_surge',
    'rolling_avg_sales', 'sales_vs_reorder', 'demand_to_stock_ratio'
]

# Step 6: Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # to include 'promotion_applied' and 'holiday_indicator'
)

# Step 7: Build pipeline with LightGBM
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42))
])

# Step 8: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.26,random_state=642)

# OPTIONAL: GridSearch for hyperparameter tuning
# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'classifier__n_estimators': [100, 200],
#     'classifier__max_depth': [5, 10],
#     'classifier__learning_rate': [0.05, 0.1]
# }
# grid = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='accuracy')
# grid.fit(X_train, y_train)
# model_pipeline = grid.best_estimator_

# Step 9: Train and evaluate
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 10: Save model
joblib.dump(model_pipeline, 'stockout_model.pkl')
print("✅ Model saved as 'stockout_model.pkl'")
