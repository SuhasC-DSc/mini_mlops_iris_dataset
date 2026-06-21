from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

#Loading Dataset
iris = load_iris()

#Training Dataset
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)
rf=RandomForestClassifier()
rf.fit(X_train,y_train)

#Save model
joblib.dump(rf, "model.pkl")

#Feature name for final output
target=iris.target_names
Target=target.tolist()

#Loading saved model
model=joblib.load("model.pkl")

#Predicting for new value
try:
    new_value = [float(input(f"{feature}: ")) for feature in iris.feature_names]
except ValueError:
    print("Please enter numeric values only.")

#Output
output=model.predict([new_value])
print("Iris plant type:",Target[output[0]])