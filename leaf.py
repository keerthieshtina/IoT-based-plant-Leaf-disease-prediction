import pandas as pd
d=pd.read_csv("plant_disease_data.csv")
print(d.info())
d=d.drop("Plant_ID",axis=1)
print(d)
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
mm=MinMaxScaler()
d["Leaf_Spot_Size"]=mm.fit_transform(d[["Leaf_Spot_Size"]])
d["Humidity"]=mm.fit_transform(d[["Humidity"]])
d["Temperature"]=mm.fit_transform(d[["Temperature"]])
print(d)
d.to_csv("new_leaf_data.csv")
