
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download
import pickle

# Load model
model_path = hf_hub_download(
    repo_id="kalrap/predictive-maintenance-model",
    filename="best_engine_model.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)

def preprocess_input(data):
    return pd.DataFrame([data])

def predict(engine_rpm, lub_oil_pressure, fuel_pressure,
            coolant_pressure, lub_oil_temp, coolant_temp):

    data = {
        "engine_rpm": engine_rpm,
        "lub_oil_pressure": lub_oil_pressure,
        "fuel_pressure": fuel_pressure,
        "coolant_pressure": coolant_pressure,
        "lub_oil_temp": lub_oil_temp,
        "coolant_temp": coolant_temp
    }

    df = preprocess_input(data)
    pred = model.predict(df)[0]

    return "Fault Detected" if pred == 1 else "Normal Engine"

interface = gr.Interface(
    fn=predict,
    inputs=["number","number","number","number","number","number"],
    outputs="text",
    title="Predictive Maintenance Model"
)

interface.launch()
