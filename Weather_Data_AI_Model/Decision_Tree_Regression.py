import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#loading the dataset
data = pd.read_excel('DQN1 Dataset.xlsx', sheet_name='Data')
# keep only the relevant columns and set the target column
features_col = ["pm2.5", "co2"]
target_col = ["healthRiskScore"]
#set variables for features and target
X = data[features_col]
y = data[target_col]
#print to verify the data
print(X.head())
print(y.head())

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create the decision tree regressor model
my_tree_reg = DecisionTreeRegressor(
    criterion='squared_error',
    splitter='best',
    max_depth=None
)
#Train the model on the training data
my_tree_reg.fit(X_train, y_train)

#part D - evaluate the model on the test set

# make predictions on the test set
y_pred = my_tree_reg.predict(X_test)
# calculate mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
#print results to console
print("RMSE:", mse)
print("R-squared:", r2)

#Task 2:
#grid search for hyperparameter tuning
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=my_tree_reg, 
    param_grid=param_grid,
    scoring = 'neg_mean_squared_error',
    cv=5, n_jobs=-1, verbose=2
)
#fit the grid search to the training data
grid_search.fit(X_train, y_train)
#print the best hyperparameters and the best score
best_grid_tree = grid_search.best_estimator_
print("Best Hyperparameters Grid Search:", grid_search.best_params_)
print("Best Score (Negative MSE):", grid_search.best_score_)

#begin the random search for hyperparameter tuning
param_dist={
    "max_depth": randint(1, 50),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20)
}

random_search = RandomizedSearchCV(
    estimator=my_tree_reg,
    param_distributions=param_dist,
    n_iter=100,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit the random search to the training data
random_search.fit(X_train, y_train)

# Print the best hyperparameters and the best score from random search
best_random_tree = random_search.best_estimator_
print("Best Hyperparameters (Random Search):", random_search.best_params_)
print("Best Score (Random Search - Negative MSE):", random_search.best_score_)

#Regularization techniques for decision tree regression
#limiting the depth of the tree
my_tree_reg_limited = DecisionTreeRegressor(
    max_depth=10,  # limit the depth of the tree
    random_state=42
)
my_tree_reg_limited.fit(X_train, y_train)

#Assigning minimum samples to each leaf node
my_tree_reg_min_samples = DecisionTreeRegressor(
    max_depth=None,  # allow the tree to grow until all leaves are pure
    min_samples_leaf=5,  # require at least 5 samples in each leaf node
    random_state=42
)
my_tree_reg_min_samples.fit(X_train, y_train)

#print the results of the regularized models
y_pred_limited = my_tree_reg_limited.predict(X_test)
y_pred_min_samples = my_tree_reg_min_samples.predict(X_test)
print("RMSE (Limited Depth):", mean_squared_error(y_test, y_pred_limited))
print("RMSE (Min Samples Leaf):", mean_squared_error(y_test, y_pred_min_samples))

#Ensemble methods for decision tree regression
#Random Forest Regressor
rf_reg = RandomForestRegressor(
    n_estimators=200,  # number of trees in the forest
    max_depth=None,
    min_samples_leaf=2,# allow trees to grow until all leaves are pure
    random_state=42,
    n_jobs=-1)
#fit the random forest regressor to the training data
rf_reg.fit(X_train, y_train.values.ravel())

#add gradient boosting regressor
gb_reg = GradientBoostingRegressor(
    n_estimators=200,  # number of boosting stages
    learning_rate=0.05,  # learning rate shrinks the contribution of each tree
    max_depth=3,  # limit the depth of the individual trees
    random_state=42
)
#fit the gradient boosting regressor to the training data
gb_reg.fit(X_train, y_train.values.ravel())

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> RMSE: {mse:.4f}, R-squared: {r2:.4f}")

evaluate_model("Original Decision Tree", my_tree_reg, X_test, y_test)

# grid search optimized decision tree
evaluate_model("Grid Search Optimized Decision Tree", best_grid_tree, X_test, y_test)

# random search optimized decision tree
evaluate_model("Random Search Optimized Decision Tree", best_random_tree, X_test, y_test)

# regularized trees
evaluate_model("Limited Depth Tree", my_tree_reg_limited, X_test, y_test)
evaluate_model("Min Samples Leaf Tree", my_tree_reg_min_samples, X_test, y_test)

# ensembles
evaluate_model("Random Forest", rf_reg, X_test, y_test)
evaluate_model("Gradient Boosting", gb_reg, X_test, y_test)