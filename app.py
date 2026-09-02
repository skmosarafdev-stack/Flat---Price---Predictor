from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Model load
model = joblib.load('model/flat_price_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        area = float(request.form['Area_sqft'])
        facing = request.form['Facing']
        floor = int(request.form['Floor'])
        parking = float(request.form['Parking_sqft'])
        bedrooms = int(request.form['Bedrooms'])

        # Simple encoding for Facing
        facing_map = {'North':0, 'South':1, 'East':2, 'West':3}
        facing_val = facing_map.get(facing, 0)

        data = [[area, facing_val, floor, parking, bedrooms]]
        pred = model.predict(data)[0]
        prediction = f"{pred:.2f} Lakh"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
