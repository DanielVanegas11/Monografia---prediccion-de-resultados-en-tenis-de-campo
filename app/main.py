from fastapi import FastAPI


app = FastAPI()


@app.get("/model_prediction")
async def model_prediction():
    pass

