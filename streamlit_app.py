import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from mistralai import Mistral
from PIL import Image
import io
import fitz  # PyMuPDF
import base64
import json

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "mistral.env")
load_dotenv(env_path)

# Try getting key from environment (local .env) or Streamlit secrets (cloud)
api_key = os.getenv("mistral_api_key")
if not api_key and "mistral_api_key" in st.secrets:
    api_key = st.secrets["mistral_api_key"]

# Initialize Mistral Client
if not api_key:
    st.error("Mistral API Key not found. Please check mistral.env file or Streamlit Secrets.")
    st.stop()

client = Mistral(api_key=api_key)

# Page Config
st.set_page_config(
    page_title="Driver Onboarding Platform (DOP)",
    page_icon="qh",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .stHeader {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚛 Driver Onboarding Platform (DOP)")
st.markdown("---")

# Sidebar for Driver Details
with st.sidebar:
    st.header("Driver Details")
    driver_name = st.text_input("Full Name", placeholder="John Doe")
    driver_phone = st.text_input("Phone Number", placeholder="+1 123 456 7890")
    driver_email = st.text_input("Email Address", placeholder="john@example.com")
    
    st.markdown("---")
    st.header("Document Upload")
    
    # CDL Upload
    st.subheader("CDL Document")
    cdl_source = st.radio("Select Input Method for CDL", ["Upload File", "Camera"], key="cdl_source")
    cdl_file = None
    if cdl_source == "Upload File":
        cdl_file = st.file_uploader("Upload CDL (Image/PDF)", type=['png', 'jpg', 'jpeg', 'pdf'], key="cdl_upload")
    else:
        cdl_file = st.camera_input("Take a picture of CDL", key="cdl_camera")

    # Medical Card Upload
    st.subheader("Medical Card")
    med_source = st.radio("Select Input Method for Medical Card", ["Upload File", "Camera"], key="med_source")
    med_file = None
    if med_source == "Upload File":
        med_file = st.file_uploader("Upload Medical Card (Image/PDF)", type=['png', 'jpg', 'jpeg', 'pdf'], key="med_upload")
    else:
        med_file = st.camera_input("Take a picture of Medical Card", key="med_camera")

# Helper Functions
def process_file(uploaded_file):
    """Converts uploaded file (PDF/Image) to base64 encoded image string."""
    if uploaded_file is None:
        return None, None
    
    file_type = uploaded_file.type
    
    if "pdf" in file_type:
        # Convert PDF to Image (First page only for now)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        return image, base64.b64encode(img_data).decode('utf-8')
    else:
        # Image file
        image = Image.open(uploaded_file)
        # Convert to RGB if RGBA
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return image, base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_data_with_mistral(base64_image, doc_type):
    """Sends image to Mistral for OCR extraction."""
    
    # Define fields based on document type
    if doc_type == "CDL":
        prompt = """
        You are an expert OCR system. Analyze the provided image of a CDL (Commercial Driver's License).
        Extract the following fields and return them in a strict JSON format with the key "CDL".
        
        Fields to extract:
        - first_name: string
        - last_name: string
        - date_of_birth: date (YYYY-MM-DD)
        - expiry_date: date (YYYY-MM-DD)
        - state_issued_cdl: string
        - license_number: alphanumeric string
        - address: string

        Example Output:
        {
            "CDL": {
                "first_name": "John",
                "last_name": "Doe",
                "date_of_birth": "1985-06-12",
                "expiry_date": "2028-06-12",
                "state_issued_cdl": "TX",
                "license_number": "CDL1234567",
                "address": "1234 Elm Street, Dallas, TX"
            }
        }
        """
    else: # Medical Card
        prompt = """
        You are an expert OCR system. Analyze the provided image of a Medical Card.
        Extract the following fields and return them in a strict JSON format with the key "Medical_Card".
        
        Fields to extract:
        - first_name: string
        - last_name: string
        - stamped_verification: string
        - expiration_date: datetime (YYYY-MM-DDTHH:MM:SS)

        Example Output:
        {
            "Medical_Card": {
                "first_name": "John",
                "last_name": "Doe",
                "stamped_verification": "Verified by Dr. Smith",
                "expiration_date": "2025-09-30T00:00:00"
            }
        }
        """

    prompt += "\nDo not include any markdown formatting (like ```json). Just return the raw JSON object. If a field is not found, set it to null."

    try:
        chat_response = client.chat.complete(
            model="pixtral-12b-2409",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ]
                }
            ]
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling Mistral API: {e}")
        return None

# Main Content Area
col1, col2 = st.columns([1, 1])

# Database File Path
DB_FILE = "driver_database.csv"
IMAGES_DIR = "uploaded_images"

# Ensure images directory exists
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

with col1:
    st.subheader("Document Preview")
    
    preview_col1, preview_col2 = st.columns(2)
    
    cdl_image, cdl_base64 = process_file(cdl_file)
    with preview_col1:
        if cdl_image:
            st.image(cdl_image, caption="CDL Preview", use_container_width=True)
            with st.expander("View CDL"):
                st.image(cdl_image)

    med_image, med_base64 = process_file(med_file)
    with preview_col2:
        if med_image:
            st.image(med_image, caption="Medical Card Preview", use_container_width=True)
            with st.expander("View Medical Card"):
                st.image(med_image)

with col2:
    st.subheader("Extraction Results")
    
    if st.button("Save & Extract Data"):
        if not driver_name or not driver_phone:
            st.warning("Please enter Driver Name and Phone Number.")
        elif not cdl_base64 and not med_base64:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Extracting data and saving..."):
                
                # Initialize driver data dictionary
                current_driver_data = {
                    "Full Name": driver_name,
                    "Phone Number": driver_phone,
                    "Email": driver_email,
                    "CDL_first_name": None,
                    "CDL_last_name": None,
                    "CDL_date_of_birth": None,
                    "CDL_expiry_date": None,
                    "CDL_state_issued_cdl": None,
                    "CDL_license_number": None,
                    "CDL_address": None,
                    "Medical_first_name": None,
                    "Medical_last_name": None,
                    "Medical_stamped_verification": None,
                    "Medical_expiration_date": None,
                    "CDL_Image_Path": None,
                    "Medical_Image_Path": None
                }

                # Process CDL
                if cdl_base64:
                    # Save Image
                    cdl_filename = f"{driver_phone}_CDL.jpg"
                    cdl_path = os.path.join(IMAGES_DIR, cdl_filename)
                    cdl_image.save(cdl_path)
                    current_driver_data["CDL_Image_Path"] = cdl_path

                    # Extract Data
                    cdl_json_str = extract_data_with_mistral(cdl_base64, "CDL")
                    if cdl_json_str:
                        try:
                            cdl_json_str = cdl_json_str.replace("```json", "").replace("```", "").strip()
                            data = json.loads(cdl_json_str)
                            cdl_data = data.get("CDL", data)
                            
                            # Map to prefixed keys
                            for key, value in cdl_data.items():
                                current_driver_data[f"CDL_{key}"] = value
                        except json.JSONDecodeError:
                            st.error("Failed to parse CDL JSON response.")

                # Process Medical Card
                if med_base64:
                    # Save Image
                    med_filename = f"{driver_phone}_Medical.jpg"
                    med_path = os.path.join(IMAGES_DIR, med_filename)
                    med_image.save(med_path)
                    current_driver_data["Medical_Image_Path"] = med_path

                    # Extract Data
                    med_json_str = extract_data_with_mistral(med_base64, "Medical Card")
                    if med_json_str:
                        try:
                            med_json_str = med_json_str.replace("```json", "").replace("```", "").strip()
                            data = json.loads(med_json_str)
                            med_data = data.get("Medical_Card", data)
                            
                            # Map to prefixed keys
                            for key, value in med_data.items():
                                current_driver_data[f"Medical_{key}"] = value
                        except json.JSONDecodeError:
                            st.error("Failed to parse Medical Card JSON response.")

                # Save to CSV
                df_new = pd.DataFrame([current_driver_data])
                
                if os.path.exists(DB_FILE):
                    df_existing = pd.read_csv(DB_FILE)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    df_combined = df_new
                
                df_combined.to_csv(DB_FILE, index=False)
                st.success("Driver data saved successfully!")

# Display Database
st.markdown("---")
st.header("Driver Database")

if os.path.exists(DB_FILE):
    df_db = pd.read_csv(DB_FILE)
    
    # Display Database without Image Paths
    df_display = df_db.drop(columns=["CDL_Image_Path", "Medical_Image_Path"], errors='ignore')
    st.dataframe(df_display)

    # Image Viewer Section
    st.markdown("### Driver Image Viewer")
    
    # Create a selection list: "Name - Phone"
    driver_options = df_db.apply(lambda x: f"{x['Full Name']} - {x['Phone Number']}", axis=1).tolist()
    selected_driver_str = st.selectbox("Select Driver to View Images", ["Select a Driver"] + driver_options)

    if selected_driver_str != "Select a Driver":
        # Find the row
        selected_phone = selected_driver_str.split(" - ")[-1]
        driver_row = df_db[df_db['Phone Number'].astype(str) == selected_phone].iloc[0]
        
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown("**CDL Image**")
            cdl_path = driver_row.get("CDL_Image_Path")
            if pd.notna(cdl_path) and os.path.exists(cdl_path):
                st.image(cdl_path, use_container_width=True)
            else:
                st.info("No CDL Image available.")
        
        with img_col2:
            st.markdown("**Medical Card Image**")
            med_path = driver_row.get("Medical_Image_Path")
            if pd.notna(med_path) and os.path.exists(med_path):
                st.image(med_path, use_container_width=True)
            else:
                st.info("No Medical Card Image available.")
        
        st.markdown("### Driver Data")
        # Display the specific row for this driver, excluding image paths
        driver_row_display = driver_row.drop(labels=["CDL_Image_Path", "Medical_Image_Path"], errors='ignore')
        st.dataframe(pd.DataFrame([driver_row_display]))

    # CSV Export (using df_display)
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Database as CSV",
        data=csv,
        file_name='driver_database.csv',
        mime='text/csv',
    )
else:
    st.info("No data in database yet.")

