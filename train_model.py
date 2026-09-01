
import pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
df=pd.read_csv(ROOT/"data/flats.csv")
X=df[["Area_sqft","Facing","Floor","Parking_sqft","Bedrooms"]]; y=df["Price_Lakhs"]
pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),["Facing"]),
                       ("num","passthrough",["Area_sqft","Floor","Parking_sqft","Bedrooms"])])
pipe=Pipeline([("preprocessor",pre),("model",RandomForestRegressor(n_estimators=300,max_depth=8,random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
mae=mean_absolute_error(yte,pred); rmse=mean_squared_error(yte,pred)**.5; r2=r2_score(yte,pred)
joblib.dump(pipe,ROOT/"model/flat_price_model.joblib")
plt.figure(figsize=(7,5)); plt.scatter(yte,pred); mn=min(yte.min(),pred.min()); mx=max(yte.max(),pred.max()); plt.plot([mn,mx],[mn,mx],"--")
plt.xlabel("Actual Price (Lakhs)"); plt.ylabel("Predicted Price (Lakhs)"); plt.title("Actual vs Predicted Flat Prices"); plt.tight_layout(); plt.savefig(ROOT/"model/actual_vs_predicted.png",dpi=160); plt.close()
(ROOT/"model/metrics.txt").write_text(f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR2={r2:.3f}\nTest samples={len(yte)}\n")
print(mae,rmse,r2)
