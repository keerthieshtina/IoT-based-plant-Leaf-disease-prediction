import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
#from sklearn.metrics import r2_score
d=pd.read_csv("plant_disease_data.csv")
print(d.info())
d=d.drop("Plant_ID",axis=1)
print(d)
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
le=LabelEncoder()
mm=MinMaxScaler()
d=pd.get_dummies(d,columns=["Plant_Type"])
d=pd.get_dummies(d,columns=["Leaf_Color"])
d["Disease_Status"]=le.fit_transform(d["Disease_Status"])
d[["Leaf_Spot_Size", "Humidity", "Temperature"]] = mm.fit_transform(
       d[["Leaf_Spot_Size", "Humidity", "Temperature"]])
print(d)
x=d[["Leaf_Spot_Size","Humidity","Temperature"]]
y=d["Disease_Status"]
x_train=mm.fit_transform(x)
y_train=le.fit_transform(y)
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)      #testing3
model=LogisticRegression()     # algorithm
model.fit(x_train,y_train)     #training
#y_pre=model.predict(x_test)
#result=r2_score(y_test,y_pre)
#print(result)
joblib.dump(model,"518.pkl")       
joblib.dump(mm,"507.pkl")
joblib.dump(x.columns.tolist(),"535.pkl")
joblib.dump(le,"label_encoder.pkl")