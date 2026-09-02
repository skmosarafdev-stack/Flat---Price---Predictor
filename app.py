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
        try:
            area = float(request.form.get('Area_sqft') or request.form.get('area') or 0)
            facing = request.form.get('Facing') or request.form.get('facing') or 'North'
            floor = int(request.form.get('Floor') or request.form.get('floor') or 0)
            parking = float(request.form.get('Parking_sqft') or request.form.get('parking') or 0)
            bedrooms = int(request.form.get('Bedrooms') or request.form.get('bedrooms') or 0)

            facing_map = {'North':0, 'South':1, 'East':2, 'West':3}
            facing_val = facing_map.get(facing, 0)

            data = [[area, facing_val, floor, parking, bedrooms]]
            pred = model.predict(data)[0]
            prediction = f"{pred:.2f} Lakh"
        except Exception as e:
            prediction = f"Error: {e}"
            print(f"Error: {e}")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
