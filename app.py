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

def save_image_to_binary(image_data):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_data.astype('uint8'))
    # Save to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    # Return binary data
    return buffer.getvalue()

def connect_to_db():
    return psycopg2.connect(
        host=os.environ.get(DB_CONFIG["host_env"], DB_CONFIG["host_default"]),
        database=os.environ.get(DB_CONFIG["db_env"], DB_CONFIG["db_default"]),
        user=os.environ.get(DB_CONFIG["user_env"], DB_CONFIG["user_default"]),
        password=os.environ.get(DB_CONFIG["pass_env"], DB_CONFIG["pass_default"])
    )

def save_prediction(prediction, confidence, image_data, true_label=None):
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
                true_label INTEGER,
                image_data BYTEA
            )
        """)
        
        # Convert image to binary
        binary_image = save_image_to_binary(image_data) if image_data is not None else None
        
        # Insert prediction
        cursor.execute(
            "INSERT INTO prediction_logs (timestamp, prediction, confidence, true_label, image_data) VALUES (%s, %s, %s, %s, %s)",
            (datetime.now(), prediction, confidence, true_label, binary_image)
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
            save_prediction(st.session_state.last_prediction, st.session_state.last_confidence, canvas_result.image_data, true_label)
            st.success("Saved to database!")
        else:
            st.warning("Make a prediction first!")

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
        # Check if image_data column exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'prediction_logs' AND column_name = 'image_data'
            )
        """)
        
        has_image_column = cursor.fetchone()[0]
        
        # Add image_data column if it doesn't exist
        if not has_image_column:
            cursor.execute("""
                ALTER TABLE prediction_logs 
                ADD COLUMN image_data BYTEA
            """)
            conn.commit()
            st.info("Database schema updated to include image storage.")
        
        # Query data with or without image based on column existence
        if has_image_column:
            cursor.execute("""
                SELECT timestamp, prediction, true_label, confidence, image_data 
                FROM prediction_logs 
                ORDER BY timestamp DESC
                LIMIT 10
            """)
        else:
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
            # Extract image data for display
            timestamps = []
            predictions = []
            true_labels = []
            confidences = []
            image_data_list = []
            
            st.subheader("Prediction History")
            
            # Inject CSS for styling
            st.markdown("""
            <style>
            div.stHorizontalBlock {
                border-radius: 5px;
                padding: 5px;
                margin-bottom: 8px;
            }
            .correct-row {
                background-color: #e8f7e8;
                border: 1px solid #8eca8e;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .incorrect-row {
                background-color: #fde9e8;
                border: 1px solid #e6a5a4;
                border-radius: 5px;
                padding: 10px;
                margin-bottom: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .column-content {
                background-color: #f8f9fa;
                border-radius: 4px;
                padding: 8px;
                text-align: center;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }
            </style>
            """, unsafe_allow_html=True)
            
            for i, (timestamp, prediction, true_label, confidence, image_data) in enumerate(data):
                timestamps.append(timestamp)
                predictions.append(prediction)
                true_labels.append(true_label)
                confidences.append(confidence)
                image_data_list.append(image_data)
                
                # Create a colored card to hold the row
                style_class = "correct-row" if prediction == true_label else "incorrect-row"
                
                # Start row container with the style class
                st.markdown(f'<div class="{style_class}">', unsafe_allow_html=True)
                
                # Create columns within the div
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                
                if image_data:
                    with col1:
                        image = Image.open(io.BytesIO(image_data))
                        st.image(image, width=40)
                
                with col2:
                    st.markdown('<div class="column-content">', unsafe_allow_html=True)
                    st.markdown(f"**Prediction:** {prediction}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="column-content">', unsafe_allow_html=True)
                    st.markdown(f"**True Label:** {true_label}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    confidence_color = "#28a745" if confidence > 80 else "#ffc107" if confidence > 50 else "#dc3545"
                    st.markdown('<div class="column-content">', unsafe_allow_html=True)
                    st.markdown(f'**Confidence:** <span style="color: {confidence_color}; font-weight: bold;">{confidence:.1f}%</span>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Close the row container
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Create DataFrame for additional tabular display if needed
            # Create a list of image indicators/references instead of binary data
            image_indicators = []
            for img in image_data_list:
                if img is not None:
                    image_indicators.append("<memory at 0x{:x}>".format(id(img)))
                else:
                    image_indicators.append("None")
            
            # Add a colorful header for the table view
            st.markdown(
                """
                <div style="
                    background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                    font-weight: bold;
                    text-align: center;
                ">
                    Prediction History Data Table
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            df = pd.DataFrame({
                "Image": image_indicators,
                "Timestamp": timestamps,
                "Prediction": predictions,
                "True Label": true_labels,
                "Confidence": confidences
            })
            
            with st.expander("View as table"):
                # Add styling based on prediction accuracy
                def highlight_prediction(row):
                    if row["Prediction"] == row["True Label"]:
                        return ['background-color: #d4f7d4'] * len(row)  # Light green for correct
                    else:
                        return ['background-color: #fad6d5'] * len(row)  # Light red for incorrect
                
                # Add CSS styling for the table including borders
                styled_df = df.style.apply(highlight_prediction, axis=1).set_properties(**{
                    'border': '1px solid #e6e6e6',
                    'text-align': 'center'
                }).set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#f2f2f2'), 
                                               ('border', '1px solid #d3d3d3'),
                                               ('font-weight', 'bold'),
                                               ('text-align', 'center'),
                                               ('padding', '8px')]},
                    {'selector': 'td', 'props': [('border', '1px solid #d3d3d3'), 
                                               ('padding', '8px')]},
                    {'selector': 'tr:hover', 'props': [('background-color', '#f5f5f5')]}
                ])
                
                # Display dataframe with border container
                st.markdown('<div style="border: 2px solid #d3d3d3; border-radius: 5px; padding: 10px;">', unsafe_allow_html=True)
                st.dataframe(styled_df, use_container_width=True, height=400)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No prediction history yet.")
    
    cursor.close()
    conn.close()
except Exception as e:
    st.error(f"Error fetching history: {e}") 