import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# 1. Page Config & UI
st.set_page_config(page_title="Eco-Track AI", page_icon="♻️")
st.title("♻️ Eco-Track: AI Waste Classifier")
st.markdown("---")

# 2. Load the AI Model
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")  # This will auto-download the model on first run

model = load_yolo()

# 3. Disposal Knowledge Base
WASTE_MAP = {
    "bottle": {"category": "Plastic", "bin": "Blue Bin", "tip": "Rinse before recycling!"},
    "cup": {"category": "Disposable", "bin": "Yellow Bin", "tip": "Check if it's compostable."},
    "cell phone": {"category": "Electronic", "bin": "E-Waste Center", "tip": "Do not throw in regular trash!"},
    "laptop": {"category": "Electronic", "bin": "E-Waste Center", "tip": "Remove battery if possible."},
    "apple": {"category": "Organic", "bin": "Green Bin", "tip": "Great for composting!"}
}

# 4. Interface
img_file = st.camera_input("Scan your item")

if img_file:
    # Convert image for processing
    img = Image.open(img_file)
    img_array = np.array(img)
    
    # Run AI Detection
    results = model(img_array)
    
    found = False
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            
            if label in WASTE_MAP:
                found = True
                data = WASTE_MAP[label]
                
                # Professional Result Card
                st.success(f"Detected: **{label.capitalize()}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Category", data['category'])
                with col2:
                    st.metric("Target Bin", data['bin'])
                
                st.info(f"💡 **Eco-Tip:** {data['tip']}")
                break

    if not found:
        st.warning("Object recognized, but disposal rules aren't set for this yet. Try a bottle or phone!")
