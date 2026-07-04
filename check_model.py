from joblib import load

scaler = load("scaler.pkl")

print("Expected Features")
print(scaler.n_features_in_)

print()

print("Feature Names")
print(scaler.feature_names_in_)