import pandas as pd
import joblib
import requests
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURACION ---
OPENWEATHER_API_KEY = "c4928901293ce06a0dbf1f02d94b3b04" # <--- RECUERDA PONER TU KEY
LAT = 18.4861
LON = -69.9312

app = FastAPI(title="Pro-Hosp API", version="2.0 Future")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CARGA DE MODELOS ---
model_stay = None
model_block = None
MODELOS_OK = False

try:
    model_stay = joblib.load("model_stay_days_logistic.pkl")       
    model_block = joblib.load("model_stay_block_randomforest.pkl") 
    MODELOS_OK = True
except Exception:
    MODELOS_OK = False

# --- INPUTS ---
class PacienteInput(BaseModel):
    Hospital_type: int
    Hospital_city: int
    Hospital_region: int
    Available_Extra_Rooms_in_Hospital: int
    Bed_Grade: float
    Patient_Visitors: int
    City_Code_Patient: float
    Admission_Deposit: float
    Department: str
    Ward_Type: str
    Ward_Facility: str
    Type_of_Admission: str 
    Illness_Severity: str
    Age: str

class FechaInput(BaseModel):
    fecha: str # Formato YYYY-MM-DD

# --- LOGICA DE CLIMA Y PRONOSTICO ---

def analizar_riesgo_clima(datos):
    """ Reglas de negocio para alertas respiratorias """
    # Si no hay datos de particulas, asumimos 0 (la API de forecast free a veces no da PM2.5)
    pm25 = datos.get("pm25", 0) 
    condicion = datos.get("condicion", "").lower()
    
    # Reglas para RD:
    # 1. Si dice "dust" o "sand" es Polvo del Sahara.
    # 2. Si dice "smoke" es humo (vertedero duquesa, etc).
    # 3. Si llueve mucho ("rain", "storm"), suben virus respiratorios/dengue.
    
    if "dust" in condicion or "sand" in condicion or "smoke" in condicion or pm25 > 75:
        return {
            "nivel": "🔴 CRÍTICO",
            "mensaje": "ALERTA: Aire muy contaminado (Polvo/Humo). Se espera pico de asma y alergias.",
            "accion": "Preparar nebulizadores y refuerzo en triaje respiratorio."
        }
    elif "rain" in condicion or "thunderstorm" in condicion:
        return {
            "nivel": "🟡 MEDIO",
            "mensaje": "Pronóstico de lluvias. Posible aumento de virus gripales y accidentes.",
            "accion": "Monitorear emergencias."
        }
    else:
        return {
            "nivel": "🟢 BAJO",
            "mensaje": "Condiciones climáticas estables.",
            "accion": "Flujo estándar."
        }

def obtener_pronostico_futuro(fecha_busqueda):
    """ Busca en los proximos 5 dias """
    try:
        # Usamos el endpoint 'forecast' en lugar de 'weather'
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=3).json()
        
        lista_pronosticos = res.get("list", [])
        
        # Buscamos en la lista si existe la fecha que el usuario pidio
        clima_encontrado = None
        
        for item in lista_pronosticos:
            # La API devuelve fecha y hora ej: "2025-12-16 12:00:00"
            # Nos quedamos solo con la fecha "2025-12-16"
            fecha_item = item["dt_txt"].split(" ")[0]
            
            if fecha_item == fecha_busqueda:
                # Encontramos datos para ese dia! Tomamos este bloque (usualmente mediodia)
                clima_encontrado = {
                    "fecha": item["dt_txt"],
                    "temp": item["main"]["temp"],
                    "humedad": item["main"]["humidity"],
                    "condicion": item["weather"][0]["description"], # ej: "light rain"
                    "viento": item["wind"]["speed"]
                    # Nota: Forecast free no siempre da PM2.5, nos guiamos por la descripcion
                }
                # Intentamos buscar el mediodia (12:00) para que sea representativo, si no, nos quedamos con el primero que halle
                if "12:00:00" in item["dt_txt"]:
                    break
        
        return clima_encontrado

    except Exception as e:
        print(f"Error forecast: {e}")
        return None

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "online 2.0"}

@app.post("/predict_patient")
def predict_patient(data: PacienteInput):
    if not MODELOS_OK: raise HTTPException(status_code=500, detail="Modelos off")

    try:
        d = data.dict()
        d['Type of Admission'] = d.pop('Type_of_Admission')
        df = pd.DataFrame([d])

        dias = int(np.round(model_stay.predict(df)[0]))
        riesgo = int(model_block.predict(df)[0])
        prob = float(model_block.predict_proba(df)[0][1])

        # LOGICA DE RESPUESTA MEJORADA PARA RD
        # Interpretamos el resultado para el usuario
        # Interpretamos el resultado
        if riesgo == 1:
            estado = "🔴 RIESGO DE BLOQUEO (> 7 días)"
        else:
            estado = "🟢 FLUJO RÁPIDO (< 7 días)"

        return {
            "dias_estimados": dias, # El modelo dice 21, pero sabemos que se refiere al grupo
            "alerta": estado,
            "confianza_del_modelo": f"{round(prob * 100, 1)}%", # Cambiamos "probabilidad" por esto
            "mensaje": "Probabilidad de que el paciente ocupe la cama más de una semana."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_demand_date")
def predict_demand_date(data: FechaInput):
    """
    Recibe una fecha futura (ej: 2025-12-20) y predice demanda respiratoria
    """
    hoy = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Si la fecha es HOY, usamos la API de tiempo real (que es más precisa con contaminación)
    if data.fecha == hoy:
        # Aquí llamarías a tu función anterior de get_live_weather (la simplifiqué para el ejemplo)
        return {"mensaje": "Para el día de hoy, usa el monitor en tiempo real."}
    
    # 2. Si es FUTURO, usamos el Forecast
    clima_futuro = obtener_pronostico_futuro(data.fecha)
    
    if clima_futuro:
        analisis = analizar_riesgo_clima(clima_futuro)
        return {
            "fecha_consultada": data.fecha,
            "pronostico_clima": clima_futuro,
            "prediccion_demanda": analisis
        }
    else:
        return {
            "error": "No hay datos para esa fecha.", 
            "motivo": "La API gratuita solo permite ver 5 días en el futuro o la fecha ya pasó."
        }