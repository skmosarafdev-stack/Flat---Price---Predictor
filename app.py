from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model/flat_price_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            area = float(request.form.get('area'))
            facing = request.form.get('facing')
            floor = int(request.form.get('floor'))
            parking = float(request.form.get('parking'))
            bedrooms = int(request.form.get('bedrooms'))

            data = pd.DataFrame([[area, facing, floor, parking, bedrooms]], columns=['Area_sqft', 'Facing', 'Floor', 'Parking_sqft', 'Bedrooms'])
            pred = model.predict(data)[0]
            prediction = round(float(pred), 2)

        except Exception as e:
            error = str(e)
    return render_template('index.html', prediction=prediction, error=error)
