import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# โหลดข้อมูล
df = pd.read_csv("heartbinary.csv")

X = df.drop("target", axis=1)
y = df["target"]

# แบ่ง train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# สร้างโมเดล Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# ทดสอบ
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# บันทึกโมเดล
joblib.dump(model, "heart_nb_model.pkl")
