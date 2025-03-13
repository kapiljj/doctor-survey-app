import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the preprocessed data
df = pd.read_csv('preprocessed_data.csv')

# Define features and target
X = df[['Active Hours', 'Count of Attempts']]
y = df['Survey Attendance Likelihood']  # Assuming this column exists

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'model.pkl')
