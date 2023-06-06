#uvicorn main:app --reload
from pydantic import BaseModel
from pydantic import Field
from enum import Enum
import pandas as pd
import joblib
from fastapi import FastAPI


app = FastAPI(title="Aplicación de predicción de resultados en Tenis de Campo")


class tourney_level(Enum):
    Grand_Slam = "G"
    Masters_1000 = "M"
    Other_tournament="A"

class PredictionOut(BaseModel):
    answer: str

class TextIn(BaseModel):
  J1_2ndWon: int = Field(..., ge=0)  # ... indica que el campo es obligatorio. ge= Acotar los valores a ser mayores o iguales a 0
  J1_ace: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J1_bpFaced: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J1_df: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J1_age: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J1_ht: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J1_rank: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J2_2ndWon: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J2_ace: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J2_bpFaced: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J2_df: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J2_age: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J2_ht: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J2_rank: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J1_hand_R: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  surface_Clay: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  surface_Grass: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  surface_Hard: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  tourney_level_G: int = Field(..., ge=0, le=1, choices=[0, 1]) # Acotar los valores a ser 0 o 1
  tourney_level_M: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0
  J2_hand_R: int = Field(..., ge=0)  # Acotar los valores a ser mayores o iguales a 0




@app.post("/prediccion")

async def Prediccion_resultado(payload: TextIn): 
    """"
    Esta aplicación está diseñada para predecir el resultado de un partido de tenis de campo pasando como argumentos las estadísticas del partido y de los jugadores para así mediante un modelo determinar si el jugador ganaría "G" o perdería ("P") el partido. \n
    \n
    A continuacuón se presenta una descripción de las variables que debe recibir el modelo para realizar las predicciones. \n
    \n
    Todas las variables que contengan en la descripción "J1" corresponden a estadísticas asociadas al jugador 1, que es sobre el cual el modelo predice si gana o pierde, mientras que las descripciones que contengan "J2" corresponderán a las estadísticas del jugador 2.
    

    J1_2ndWon: Puntos ganados con el segundo servicio. \n
    J1_ace: Saques directos. \n
    J1_bpFaced: Puntos de quiebre enfrentados.\n
    J1_df: Dobles faltas.\n
    J1_age: Edad.\n
    J1_ht: Altura.\n
    J1_rank: Posición en el ranking ATP.\n
    J2_2ndWon: Puntos ganados con el segundo servicio. \n
    J2_ace: Saques directos.\n
    J2_bpFaced: Puntos de quiebre enfrentados.\n
    J2_df: Dobles faltas.\n
    J2_age: Edad.\n
    J2_ht: Altura.\n
    J2_rank: Posición en el ranking ATP.\n
    J1_hand_R: Mano hábil del jugador (Diestro=1, Zurdo=0).\n
    surface_Clay: Indicar 1 si el partido se juega en polvo de ladrillo, 0 en cualquier otro caso.\n
    surface_Grass: Indicar 1 si el partido se juega en cesped, 0 en cualquier otro caso.\n
    surface_Hard: Indicar 1 si el partido se juega en cemento, 0 en cualquier otro caso.\n
    tourney_level_G: Indicar 1 si el torneo es Grand Slam, 0 en cualquier otro caso\n
    tourney_level_M: Indicar 1 si el torneo es Masters 1000, 0 en cualquier otro caso\n
    J2_hand_R: Mano hábil del jugador (Diestro=1, Zurdo=0).\n

    """
    modelo = joblib.load('resources/model/model.joblib')
    
    datos=pd.DataFrame(dict(payload),index=[0])

    Resultado=modelo.predict(datos)


    return {"Resultado":Resultado.item(0)}