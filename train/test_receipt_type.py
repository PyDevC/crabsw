import joblib

loaded_model = joblib.load('svm_model.pkl')
loaded_vectorizer = joblib.load('vectorizer.pkl')

new_data = ["Taxi fare", "Hotel stay", "Spa"] # if your receipt says these expenses
new_data_transformed = loaded_vectorizer.transform(new_data)
predictions = loaded_model.predict(new_data_transformed)

print("Predictions for new data:", predictions)
