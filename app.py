import gradio as gr
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("model.pkl")

# Prediction function
def predict_fraud(tx_amount, tx_hour):
    try:
        # Create DataFrame with same columns used in training
        X = pd.DataFrame([[tx_amount, tx_hour]], columns=["TX_AMOUNT", "TX_HOUR"])
        prediction = model.predict(X)[0]
        return "🚨 Fraud Detected" if prediction == 1 else "✅ Legitimate Transaction"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio app
app = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Transaction Amount"),
        gr.Number(label="Transaction Hour (0-23)")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Fraud Detection System",
    description="Enter transaction details to check if they are fraudulent."
)

# if __name__ == "__main__":
#     app.launch(server_name="0.0.0.0", server_port=8080)

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=8080)



