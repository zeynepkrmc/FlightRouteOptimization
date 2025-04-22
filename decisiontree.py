from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/flights.csv')

df['AvgTicketPrice'] = (
    df['AvgTicketPrice']
    .replace('[\$,]', '', regex=True)  
    .str.replace(',', '')            
    .astype(float)                  
)

df['FlightDelay'] = df['FlightDelay'].apply(lambda x: 1 if x == 'TRUE' else 0)
df = df.dropna()

categorical_columns = df.select_dtypes(include=['object']).columns
label_encoders = {}

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le 

df = df[['DestWeather', 'OriginWeather', 'FlightDelay', 'FlightDelayMin', 'FlightDelayType']]
X = df[['DestWeather', 'OriginWeather', 'FlightDelay', 'FlightDelayMin']] 
y = df['FlightDelayType'] 
class_names = label_encoders['FlightDelayType'].classes_ 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=4,
    min_samples_leaf=5, 
    min_samples_split=10,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, model.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=10)  
cv_accuracy = cv_scores.mean()

print(f"Cross-Validation Accuracy: {cv_accuracy:.4f}")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
plt.figure(figsize=(50, 20), dpi=75)
plot_tree(
    model,
    feature_names=X.columns,
    class_names=label_encoders['FlightDelayType'].classes_,
    filled=True,
    rounded=True,
    fontsize=6,
)
plt.title("Decision Tree", fontsize=18)
plt.show()
