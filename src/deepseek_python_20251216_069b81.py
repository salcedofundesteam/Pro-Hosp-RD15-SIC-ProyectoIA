# Importaciones actualizadas y compatibles con scikit-learn 1.3+ y 1.6+
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar datos
df = pd.read_csv("host_train.csv")

# Información del dataset
df.info()

# Preprocesamiento de Stay_Days
def convert_stay_days(value):
    """
    Convierte valores como '0-10' a entero (0) o maneja valores atípicos
    """
    if isinstance(value, str):
        try:
            # Para formato '0-10', tomamos el primer número
            if '-' in value:
                return int(value.split('-')[0])
            # Si ya es un número entero como string
            else:
                return int(value)
        except (ValueError, AttributeError):
            return np.nan
    else:
        try:
            return int(value)
        except (ValueError, TypeError):
            return np.nan

df["Stay_Days"] = df["Stay_Days"].apply(convert_stay_days)

# Crear variable binaria para clasificación
df["Stay_Block"] = df["Stay_Days"].apply(lambda x: 1 if x > 7 else 0)

# Eliminar filas con valores nulos
df.dropna(inplace=True)
print(f"Filas después de eliminar nulos: {len(df)}")

# Verificar distribución de la variable objetivo binaria
print(f"Distribución de Stay_Block:\\n{df['Stay_Block'].value_counts()}")
print(f"Proporción de Stay_Block=1: {df['Stay_Block'].mean():.2%}")

# Preparar variables independientes y dependientes
X = df.drop(columns=["Stay_Days", "case_id", "Stay_Block"])
y_reg = df['Stay_Days']  # Para regresión logística (valores discretos)
y_clf = df['Stay_Block']  # Para clasificación binaria

# Variables para modelo de regresión (días discretos)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg,
    test_size=0.4,
    random_state=20,
    stratify=y_reg
)

# Variables para modelo de clasificación binaria
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf,
    test_size=0.4,
    random_state=20,
    stratify=y_clf
)

# Verificar que no haya valores nulos
print(f"Valores nulos en X_train_reg: {X_train_reg.isnull().sum().sum()}")
print(f"Valores nulos en X_train_clf: {X_train_clf.isnull().sum().sum()}")

# Definir columnas categóricas y numéricas
categorical_cols = [
    "Department",
    "Ward_Type",
    "Ward_Facility",
    "Type of Admission",
    "Illness_Severity",
    "Age"
]

numeric_cols = [
    "Hospital_type",
    "Hospital_city",
    "Hospital_region",
    "Available_Extra_Rooms_in_Hospital",
    "Bed_Grade",
    "Patient_Visitors",
    "City_Code_Patient",
    "Admission_Deposit"
]

# Preprocesador con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

# Configuración del modelo de regresión (Logistic Regression para clases discretas)
# Nota: LogisticRegression puede manejar multi-clase para valores discretos
model_regression = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',  # Para múltiples clases discretas
        solver='lbfgs'
    ))
])

# Configuración del modelo de clasificación binaria (Random Forest)
model_classification = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=10,
        min_samples_split=5
    ))
])

# Entrenar modelos
print("Entrenando modelo de regresión (días de estancia)...")
model_regression.fit(X_train_reg, y_train_reg)
print("Entrenamiento de regresión completado.")

print("Entrenando modelo de clasificación (bloque >7 días)...")
model_classification.fit(X_train_clf, y_train_clf)
print("Entrenamiento de clasificación completado.")

# Evaluación rápida
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

# Para clasificación binaria
y_pred_clf = model_classification.predict(X_test_clf)
accuracy_clf = accuracy_score(y_test_clf, y_pred_clf)
print(f"\\nEvaluación del modelo de clasificación:")
print(f"Accuracy: {accuracy_clf:.4f}")
print(f"\\nReporte de clasificación:\\n{classification_report(y_test_clf, y_pred_clf)}")

# Para regresión (usando accuracy ya que son clases discretas)
y_pred_reg = model_regression.predict(X_test_reg)
# Convertir predicciones a enteros para comparación
y_pred_reg_int = np.round(y_pred_reg).astype(int)
accuracy_reg = accuracy_score(y_test_reg, y_pred_reg_int)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
print(f"\\nEvaluación del modelo de regresión (días):")
print(f"Accuracy (valores redondeados): {accuracy_reg:.4f}")
print(f"MSE: {mse_reg:.4f}")

# Guardar modelos
joblib.dump(model_regression, "model_stay_days_logistic.pkl")
joblib.dump(model_classification, "model_stay_block_randomforest.pkl")
print("\\nModelos guardados exitosamente.")

# Función para predecir nuevos pacientes
def predict_patient(data: dict):
    """
    Predice días de estancia y si será mayor a 7 días para un nuevo paciente
    
    Args:
        data: Diccionario con las características del paciente
        
    Returns:
        Diccionario con las predicciones
    """
    try:
        # Cargar modelos
        model_stay = joblib.load("model_stay_days_logistic.pkl")
        model_block = joblib.load("model_stay_block_randomforest.pkl")
        
        # Convertir a DataFrame
        input_df = pd.DataFrame([data])
        
        # Asegurar que las columnas estén en el orden correcto
        expected_columns = X_train_reg.columns.tolist()
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)
        
        # Realizar predicciones
        stay_prediction = int(np.round(model_stay.predict(input_df)[0]))
        block_prediction = model_block.predict(input_df)[0]
        block_probability = model_block.predict_proba(input_df)[0][1]
        
        return {
            "stay_days_prediction": stay_prediction,
            "block_prediction": int(block_prediction),
            "block_probability": float(block_probability),
            "interpretation": f"El paciente probablemente estará {stay_prediction} días. " +
                             f"Probabilidad de estancia >7 días: {block_probability:.1%}"
        }
    
    except FileNotFoundError as e:
        return {"error": f"Modelo no encontrado: {str(e)}"}
    except Exception as e:
        return {"error": f"Error en la predicción: {str(e)}"}

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de datos de un paciente
    sample_patient = {
        "Hospital": 1,
        "Hospital_type": 2,
        "Hospital_city": 3,
        "Hospital_region": 2,
        "Available_Extra_Rooms_in_Hospital": 10,
        "Department": "gynecology",
        "Ward_Type": "R",
        "Ward_Facility": "F",
        "Bed_Grade": 2.0,
        "patientid": 12345,
        "City_Code_Patient": 7.0,
        "Type of Admission": "Urgent",
        "Illness_Severity": "Moderate",
        "Patient_Visitors": 2,
        "Age": "21-30",
        "Admission_Deposit": 5000.0
    }
    
    print("\\n--- Ejemplo de predicción ---")
    result = predict_patient(sample_patient)
    print(f"Resultado: {result}")