import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# 1. Page Configuration & Professional UI Header
st.set_page_config(page_title="Eco-Track AI", page_icon="♻️")
st.title("♻️ Eco-Track: AI Waste Classifier")
st.markdown("""
    Welcome to Eco-Track! This AI-powered tool helps you sort your waste correctly 
    using real-time computer vision. Simply scan an item to see its recycling category.
""")
st.markdown("---")

# 2. Optimized AI Model Loading
@st.cache_resource
def load_yolo():
    # Downloads the lightweight YOLOv8n model for high-speed performance
    return YOLO("yolov8n.pt") 

model = load_yolo()

# 3. Disposal Knowledge Base
WASTE_MAP = {
    "bottle": {"category": "Plastic/Glass", "bin": "Blue Bin", "tip": "Rinse before recycling!"},
    "cup": {"category": "Disposable", "bin": "Yellow Bin", "tip": "Check for compostable labels."},
    "cell phone": {"category": "Electronic", "bin": "E-Waste Center", "tip": "Do not throw in regular trash!"},
    "laptop": {"category": "Electronic", "bin": "E-Waste Center", "tip": "Ensure the battery is removed."},
    "apple": {"category": "Organic", "bin": "Green Bin", "tip": "Great for your home compost!"},
    "orange": {"category": "Organic", "bin": "Green Bin", "tip": "Compostable waste."}
}

# 4. User Interface - Camera Input
img_file = st.camera_input("Scan your item using your webcam")

if img_file:
    # Convert uploaded image for AI processing
    img = Image.open(img_file)
    img_array = np.array(img)
    
    # Run AI Detection (Inference)
    results = model(img_array)
    
    found = False
    for r in results:
        for box in r.boxes:
            # Identify the object name from the model
            label = model.names[int(box.cls[0])]
            
            if label in WASTE_MAP:
                found = True
                data = WASTE_MAP[label]
                
                # Professional Result Display
                st.success(f"Detected Object: **{label.capitalize()}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Material Category", data['category'])
                with col2:
                    st.metric("Correct Disposal Bin", data['bin'])
                
                st.info(f"💡 **Sustainability Tip:** {data['tip']}")
                break

    if not found:
        st.warning("Object recognized, but disposal rules aren't set for this item yet. Try a bottle, cup, or phone!")
