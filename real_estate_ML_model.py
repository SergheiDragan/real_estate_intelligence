import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import xgboost

# Import data
real_estate_df = pd.read_csv('real_estate_data.csv')

# Selecting categorical columns
object_attributes = [cat for cat in real_estate_df.columns if real_estate_df[cat].dtypes == 'object']
print(object_attributes)

# Creating an instance of LabelEncoder class
le = LabelEncoder()
# Applying LabelEncoder to categorical columns
for cat in object_attributes:
    real_estate_df[cat] = le.fit_transform(real_estate_df[cat])

#save label encoder classes
np.save('label_encoder_classes.npy', le.classes_)

# Separate target variable from predictors
X = real_estate_df.drop('price_EUR_sqm', axis=1)
Y = real_estate_df['price_EUR_sqm']

# Split the data into train, validation and test with a ratio of 68/12/20
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.15, random_state=42)

# Checking the shape of splitted data sets
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)

print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# Import necessary packages
from xgboost import XGBRegressor

# Initialize the XGBoost regressor model
xgb_model = XGBRegressor(subsample=1, n_estimators=1000, max_depth=9, learning_rate=0.1, gamma=0.5, colsample_bytree=0.7, random_state=42)

# Fit the XGBoost regressor model on the training data
xgb_model.fit(X_train, Y_train)

# Evaluate the model on the training data
Y_train_pred = xgb_model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
train_r2 = r2_score(Y_train, Y_train_pred)

# Evaluate the model on the validation data
Y_val_pred = xgb_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(Y_val, Y_val_pred))
val_r2 = r2_score(Y_val, Y_val_pred)

print(f"XGBoost Regressor - Training RMSE: {train_rmse:.2f}, Training R^2: {train_r2:.2f}")
print(f"XGBoost Regressor - Validation RMSE: {val_rmse:.2f}, Validation R^2: {val_r2:.2f}")

# Predict test targets using the last developed model
Y_test_pred = xgb_model.predict(X_test)

# Evaluate the performance of the model on the held-out test set
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
test_r2 = r2_score(Y_test, Y_test_pred)
print('XGBoost Regressor - Test RMSE: {:.2f}, Test R^2: {:.2f}'.format(test_rmse, test_r2))

# save the model as a compressed file
joblib.dump(xgb_model, 'best_model.joblib.gz', compress=('gzip', 3))