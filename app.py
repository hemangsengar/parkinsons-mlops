import gradio as gr
import joblib  # use joblib since your model was saved with it

# Load the model
model = joblib.load("model.pkl")  # make sure model.pkl is in root

def predict(input_data):
    try:
        # Convert comma-separated string to list of floats
        input_list = [float(x.strip()) for x in input_data.split(",")]
        
        # Predict
        result = model.predict([input_list])[0]
        
        # Map result to human-readable label
        if result == 0:
            return "0 (No Parkinson’s detected)"
        else:
            return "1 (Parkinson’s detected)"
    
    except Exception as e:
        return f"Error in prediction: {e}"


# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter input data"),  # change input type if needed
    outputs=gr.Textbox(label="Prediction"),
    title="Parkinson's Detection",
    description="Enter your data to detect Parkinson's"
)

iface.launch()

