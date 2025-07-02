import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import joblib

# Load data
df = pd.read_csv('Walmart.csv')

# Create new target for regression
# Avoid divide-by-zero and add smoothing for more variation
df['sales_per_day'] = df['quantity_sold'] / 7  # Assuming weekly data
df['sales_per_day'] = df['sales_per_day'].replace(0, 0.1)  # Avoid infinite
df['days_until_stockout'] = df['inventory_level'] / df['sales_per_day']
df['days_until_stockout'] = df['days_until_stockout'].clip(upper=30)  # Cap to prevent regression issues
df['days_until_stockout'] = df['days_until_stockout'].clip(upper=30)  # Cap outliers

# Define features
features = [
    'inventory_level', 'reorder_point', 'reorder_quantity',
    'quantity_sold', 'forecasted_demand', 'actual_demand',
    'supplier_lead_time', 'promotion_applied', 'promotion_type',
    'holiday_indicator', 'weather_conditions', 'weekday', 'store_location'
]

target_class = 'stockout_indicator'
target_days = 'days_until_stockout'

X = df[features]
y_class = df[target_class]
y_days = df[target_days]

# Preprocessing
categorical = ['promotion_type', 'weather_conditions', 'weekday', 'store_location']
numerical = ['inventory_level', 'reorder_point', 'reorder_quantity',
             'quantity_sold', 'forecasted_demand', 'actual_demand', 'supplier_lead_time']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder='passthrough')

# Pipelines
classification_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(random_state=42))
])

regression_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(random_state=42))
])

# Split and Train
X_train, X_test, y_class_train, y_class_test, y_days_train, y_days_test = train_test_split(
    X, y_class, y_days, test_size=0.2, random_state=42
)

classification_model.fit(X_train, y_class_train)
regression_model.fit(X_train, y_days_train)

# Save both models
joblib.dump(classification_model, 'stockout_classifier.pkl')
joblib.dump(regression_model, 'stockout_regressor.pkl')

print("✅ Models saved: 'stockout_classifier.pkl' and 'stockout_regressor.pkl'")
