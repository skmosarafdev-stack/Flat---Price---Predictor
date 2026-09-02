from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Model load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

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
        prediction = f"{pred:,.2f} Lakh"

    return render_template('index.html', prediction=prediction)

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    results = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            # Encode Facing if needed
            if 'Facing' in df.columns and df['Facing'].dtype == 'object':
                facing_map = {'North':0, 'South':1, 'East':2, 'West':3}
                df['Facing'] = df['Facing'].map(facing_map)

            preds = model.predict(df)
            df['Predicted_Price_Lakh'] = preds
            results = df.to_html(classes='table table-striped', index=False)

    return render_template('batch.html', results=results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
