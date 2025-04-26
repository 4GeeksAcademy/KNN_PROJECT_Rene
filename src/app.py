from utils import db_connect
engine = db_connect()



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

# 1. Cargar el dataset
url = 'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/winequality-red.csv'
df = pd.read_csv(url, sep=';')

# 2. Separar features (X) y target (y)
X = df.drop('quality', axis=1)
y = df['quality']

# 3. División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Escalado de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entrenamiento inicial con K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# 6. Evaluación inicial
print("🔍 Resultados iniciales con K=5")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 7. Optimización del parámetro K
k_values = list(range(1, 21))
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# 8. Gráfico Accuracy vs K
plt.figure(figsize=(10,6))
plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs K en KNN')
plt.xlabel('Número de vecinos (K)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.savefig('accuracy_vs_k.png')
plt.close()

# 9. Entrenamiento final con mejor K (k=1 según resultados)
final_knn = KNeighborsClassifier(n_neighbors=1)
final_knn.fit(X_train_scaled, y_train)

# 10. Guardar modelo entrenado
joblib.dump(final_knn, 'modelo_knn_vino.pkl')
print("Modelo final entrenado y guardado como 'modelo_knn_vino.pkl'")
print("Gráfico guardado como 'accuracy_vs_k.png'")
