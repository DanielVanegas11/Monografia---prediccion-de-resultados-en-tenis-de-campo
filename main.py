#uvicorn main:app --reload
from pydantic import BaseModel,constr,conint,StrictInt
from pydantic import Field
from enum import Enum
import pandas as pd
import joblib
from fastapi import FastAPI


app = FastAPI(title="Aplicación de predicción de resultados en Tenis de Campo")
    
class PredictionOut(BaseModel):
    answer: str

class TextIn(BaseModel):
  J1_2ndWon: conint(ge=0, le=100, strict=True) = Field(...,)
  J1_ace: conint(ge=0, le=100, strict=True) = Field(...,)
  J1_bpFaced:  conint(ge=0, le=100, strict=True) = Field(...,)
  J1_df:  conint(ge=0, le=100, strict=True) = Field(...,)
  J1_age:  conint(ge=10, le=100, strict=True) = Field(...,)
  J1_ht:  conint(ge=100, le=250, strict=True) = Field(...,)
  J1_rank:  conint(ge=1, le=100, strict=True) = Field(...,)
  J2_2ndWon:  conint(ge=0, le=100, strict=True) = Field(...,)
  J2_ace:  conint(ge=0, le=100, strict=True) = Field(...,)
  J2_bpFaced:  conint(ge=0, le=100, strict=True) = Field(...,)
  J2_df:conint(ge=0, le=100, strict=True) = Field(...,)
  J2_age: conint(ge=10, le=100, strict=True) = Field(...,)
  J2_ht:  conint(ge=100, le=250, strict=True) = Field(...,)
  J2_rank:   conint(ge=1, le=100, strict=True) = Field(...,)
  J1_hand_R:   conint(ge=0, le=1, strict=True) = Field(...,)
  surface: str = Field(..., choices=["C", "G", "H","P"])
  Tourney_level: str = Field(..., choices=["M", "G", "A"])
  J2_hand_R:   conint(ge=0, le=1, strict=True) = Field(...,)


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
    surface: Indicar "H" si es pista dura (Hard), "G" si es cesped (Grass), "C" si es polvo de ladrillo (Clay) ó "P" si es carpeta (Carpet). \n
    Tourney_level: Indicar "G" si es Grand Slam, "M" si es Masters 1000 ó "A" si corresponde a un torneo diferente del circuito ATP.
    J2_hand_R: Mano hábil del jugador (Diestro=1, Zurdo=0).\n

    """
    modelo = joblib.load('resources/model/model.joblib')  #Se carga el modelo
    
    feature_name=['J1_2ndWon', 'J1_ace', 'J1_bpFaced', 'J1_df', 'J1_age', 'J1_ht',
       'J1_rank', 'J2_2ndWon', 'J2_ace', 'J2_bpFaced', 'J2_df', 'J2_age',
       'J2_ht', 'J2_rank', 'J1_hand_R', 'surface_Clay', 'surface_Grass',
       'surface_Hard', 'tourney_level_G', 'tourney_level_M', 'J2_hand_R'] #Se definen las variables que alimentan el modelo en el orden que debe ser.
    
    datos = pd.DataFrame(columns=feature_name)  #Se crea un dataframe vacío con las columnas del modelo
    datos_payload = pd.DataFrame(dict(payload), index=[0]).drop(["surface","Tourney_level"],axis=1) # Se crea un dataframe con base en el diccionario generado con los valores ingresados por los usuarios sin incluir aquellas variables que no hacen parte del modelo ya que más adelante si codificará el mismo.
    datos = datos.merge(datos_payload, how='outer') #Se hace un JOIN con ambos dataframe para que así los valores ingresados por los usuarios queden en el df datos y los únicos valores que quedan vacios (n/a) corresponden a las características que deben ser codificadas ("Surface" y "Tourney_level")
    
    surface = payload.surface  
    #Se crea una variable llamada surface que contendrá el valor ingresado por el usuario para el tipo de superficie, luego se hace un condicional para asignar los valores a las características del modelo según sea en cada caso.  
    if surface=="G":
       datos['surface_Clay'].fillna(0, inplace=True)
       datos["surface_Grass"].fillna(1, inplace=True)
       datos["surface_Hard"].fillna(0, inplace=True)

    elif surface=="C":
       datos['surface_Clay'].fillna(1, inplace=True)
       datos["surface_Grass"].fillna(0, inplace=True)
       datos["surface_Hard"].fillna(0, inplace=True)

    elif surface=="H":
       datos['surface_Clay'].fillna(0, inplace=True)
       datos["surface_Grass"].fillna(0, inplace=True)
       datos["surface_Hard"].fillna(1, inplace=True)

    elif surface=="P":
       datos['surface_Clay'].fillna(0, inplace=True)
       datos["surface_Grass"].fillna(0, inplace=True)
       datos["surface_Hard"].fillna(0, inplace=True)

    Tourney_level=payload.Tourney_level
    #Se crea una variable llamada Tourney_level que contendrá el valor ingresado por el usuario para el tipo de tornep, luego se hace un condicional para asignar los valores a las características del modelo según sea en cada caso.
    if Tourney_level=="G":
       datos['tourney_level_G'].fillna(1, inplace=True)
       datos["tourney_level_M"].fillna(0, inplace=True)

    elif Tourney_level=="M":
       datos['tourney_level_G'].fillna(0, inplace=True)
       datos["tourney_level_M"].fillna(1, inplace=True)

    elif Tourney_level=="A":
       datos['tourney_level_G'].fillna(0, inplace=True)
       datos["tourney_level_M"].fillna(0, inplace=True)

    Resultado=modelo.predict(datos)
    
    return {"Resultado":Resultado.item(0)}