from flask import Flask, render_template, request
import joblib
import os

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

            facing_map = {'North':0, 'South':1, 'East':2, 'West':3}
            facing_val = facing_map.get(facing, 0)

            data = [[area, facing_val, floor, parking, bedrooms]]
            pred = model.predict(data)[0]
            prediction = float(pred) # sudhu number pathabo, Lakh html e add hobe
        except Exception as e:
            error = str(e)
            print(f"Error: {e}")

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
