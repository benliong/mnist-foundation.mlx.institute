import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import io
from streamlit_drawable_canvas import st_canvas
import psycopg2
from datetime import datetime
import os
from config import MNIST_MEAN, MNIST_STD, DB_CONFIG

# App title and header
st.set_page_config(page_title="Digit Recognizer")
st.title("Digit Recognizer")

# 1. Load the model
@st.cache_resource
def load_model():
    # Define the same model architecture as in train.py
    from torch import nn
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, 1),          # 28→26
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1),         # 26→24
                nn.ReLU(),
                nn.MaxPool2d(2),                 # 24→12
                nn.Flatten(),
                nn.Linear(64 * 12 * 12, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.net(x)
    
    model = Net()
    model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

# 2. Draw canvas for user input
st.subheader("Draw a digit (0-9)")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 3. Prediction and DB connection
def preprocess_image(image_data):
    # Convert to grayscale and resize to 28x28
    image = Image.fromarray(image_data.astype('uint8')).convert('L')
    image = image.resize((28, 28))
    
    # Apply the same transforms as during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def connect_to_db():
    return psycopg2.connect(
        host=os.environ.get(DB_CONFIG["host_env"], DB_CONFIG["host_default"]),
        database=os.environ.get(DB_CONFIG["db_env"], DB_CONFIG["db_default"]),
        user=os.environ.get(DB_CONFIG["user_env"], DB_CONFIG["user_default"]),
        password=os.environ.get(DB_CONFIG["pass_env"], DB_CONFIG["pass_default"])
    )

def save_prediction(prediction, confidence, true_label=None):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP,
                prediction INTEGER,
                confidence FLOAT,
                true_label INTEGER
            )
        """)
        
        # Insert prediction
        cursor.execute(
            "INSERT INTO prediction_logs (timestamp, prediction, confidence, true_label) VALUES (%s, %s, %s, %s)",
            (datetime.now(), prediction, confidence, true_label)
        )
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")

# 4. UI for prediction and feedback
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # Get the model
            model = load_model()
            
            # Preprocess the image from canvas
            tensor = preprocess_image(canvas_result.image_data)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                prediction = torch.argmax(probabilities).item()
                confidence = probabilities[prediction].item() * 100
            
            # Display prediction
            st.subheader(f"Prediction: {prediction}")
            st.write(f"Confidence: {confidence:.1f}%")
            
            # Store in session state for later saving
            st.session_state.last_prediction = prediction
            st.session_state.last_confidence = confidence
        else:
            st.warning("Please draw a digit first!")

with col2:
    # Get true label from user
    st.subheader("Enter true label")
    true_label = st.number_input("True label:", min_value=0, max_value=9, step=1)
    
    if st.button("Submit"):
        if 'last_prediction' in st.session_state and 'last_confidence' in st.session_state:
            # Save prediction and true label to database
            save_prediction(st.session_state.last_prediction, st.session_state.last_confidence, true_label)
            st.success("Saved to database!")
        else:
            st.warning("Make a prediction first!")

# 5. Display history from database
st.header("History")

try:
    conn = connect_to_db()
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'prediction_logs'
        )
    """)
    
    if cursor.fetchone()[0]:
        cursor.execute("""
            SELECT timestamp, prediction, true_label, confidence 
            FROM prediction_logs 
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        
        # Convert to DataFrame for display
        import pandas as pd
        data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=["timestamp", "Prediction", "True Label", "confidence"])
            
            # Add styling based on prediction accuracy
            def highlight_prediction(row):
                if row["Prediction"] == row["True Label"]:
                    return ['background-color: #d4f7d4'] * len(row)  # Light green for correct
                else:
                    return ['background-color: #fad6d5'] * len(row)  # Light red for incorrect
            
            styled_df = df.style.apply(highlight_prediction, axis=1)
            st.dataframe(styled_df)
        else:
            st.info("No prediction history yet.")
    
    cursor.close()
    conn.close()
except Exception as e:
    st.error(f"Error fetching history: {e}") 