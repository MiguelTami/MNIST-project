from features.features import generar_features
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import os


def entrenar_modelo_pipeline(
    nombre_modelo,
    ruta_features,
    ruta_etiquetas,
    modelo_final=None,
    guardar=True
):
    
    columnas = [
        "intensidad_promedio",
        "varianza",
        "densidad_superior",
        "simetria_vertical",
        "centro_masa_vertical",
        "simetria_horizontal"
    ]
    print(f"\n🧪 Preparando modelo modelo: {nombre_modelo}")

    # Generar features    
    ## 1. Cargar datos
    X_train_img = pd.read_csv(os.path.join(ruta_etiquetas, "X_train.csv"))
    X_val_img   = pd.read_csv(os.path.join(ruta_etiquetas, "X_val.csv"))
    X_test_img  = pd.read_csv(os.path.join(ruta_etiquetas, "X_test.csv"))

    ## 2. Generar features
    X_train_features = generar_features(X_train_img)
    X_val_features   = generar_features(X_val_img)
    X_test_features  = generar_features(X_test_img)

    ## 3.Guardar features
    X_train_features.to_csv(os.path.join(ruta_features, "X_train_features.csv"), index=False)
    X_val_features.to_csv(os.path.join(ruta_features, "X_val_features.csv"), index=False)
    X_test_features.to_csv(os.path.join(ruta_features, "X_test_features.csv"), index=False)

    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_val_scaled   = scaler.transform(X_val_features)
    X_test_scaled  = scaler.transform(X_test_features)   
   
    y_train = pd.read_csv(os.path.join(ruta_etiquetas, "y_train.csv"))
    y_train = y_train.squeeze()

    y_val = pd.read_csv(os.path.join(ruta_etiquetas, "y_val.csv"))
    y_val = y_val.squeeze()

    y_test = pd.read_csv(os.path.join(ruta_etiquetas, "y_test.csv"))
    y_test = y_val.squeeze()

    # 3. Modelo
    ## Entrenamiento del modelo
    print("🔍 Entrenando modelo...")
    modelo_sklearn = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    modelo_sklearn.fit(X_train_scaled, y_train)

    ## Si es el modelo final, se entrena y evalua con los datos de testeo. Sino con los de validación
    if modelo_final:
        y_pred = modelo_sklearn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
    else:
        y_pred = modelo_sklearn.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f}")

    # 4. Matriz de confusión (opcional)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title("Matriz de confusión")
    plt.show()

    # Creamos tabla de pesos
    pesos_df = pd.DataFrame(modelo_sklearn.coef_, columns=columnas)
    pesos_df.index = [f'Clase {i}' for i in range(10)]
    print(pesos_df)

    if guardar:
        joblib.dump(modelo_sklearn, f"../output/models/{nombre_modelo}.pkl")
        joblib.dump(scaler, f"../output/models/{nombre_modelo}_scaler.pkl")
