
from flask import Flask,render_template,request
import joblib
from pathlib import Path
app=Flask(__name__); model=joblib.load(Path(__file__).parent/"model/flat_price_model.joblib")
@app.route("/",methods=["GET","POST"])
def index():
    prediction=None; error=None; values={}
    if request.method=="POST":
        try:
            values={"Area_sqft":float(request.form["area"]),"Facing":request.form["facing"],
                    "Floor":int(request.form["floor"]),"Parking_sqft":float(request.form["parking"]),
                    "Bedrooms":int(request.form["bedrooms"])}
            prediction=float(model.predict([values])[0])
        except Exception: error="Please enter valid property details."
    return render_template("index.html",prediction=prediction,error=error,values=values)
if __name__=="__main__": app.run(debug=True)
