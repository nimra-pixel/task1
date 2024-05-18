from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

# Sample data representing student attributes
data = {
    'Name': ['John', 'Emily', 'Sarah', 'Michael'],
    'CGPA': [3.5, 3.8, 3.2, 3.6],
    'Marks': [85, 90, 78, 88],
    'Percentage': [82.5, 87.5, 75.0, 85.0]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Prepare features (X) and target variable (y)
X = df[['CGPA', 'Marks']]  # Features (CGPA, Marks)
y = df['Percentage']  # Target variable (Percentage)

# Initialize a linear regression model
model = LinearRegression()

# Fit the model to the entire dataset
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    
    # Prepare features for prediction
    features = pd.DataFrame(data, index=[0])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return prediction as JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run()
