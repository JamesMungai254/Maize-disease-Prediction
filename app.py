import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import streamlit as st


st.set_page_config(layout="wide")




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 4)  # Update to 4 output classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the saved model
# Use streamlit cache to load the model only once
@st.cache_resource
def load_model(path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

model = load_model('maize_disease_model.pth')

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def explain_disease(prediction):
    if prediction == 0:
        return {
            "disease": "Blight",
            "cause": "Caused by bacteria and fungi, particularly *Pseudomonas syringae* and *Xanthomonas campestris*.",
            "symptoms": "Water-soaked lesions on leaves that turn brown, stunted growth, yellowing of leaves.",
            "prevention": "Use disease-free seeds, apply fungicides, and practice crop rotation."
        }
    elif prediction == 1:
        return {
            "disease": "Common Rust",
            "cause": "Caused by the fungus *Puccinia sorghi*.",
            "symptoms": "Reddish-brown pustules on leaves, especially on the underside of the leaf.",
            "prevention": "Plant resistant maize varieties, apply fungicides, and avoid overhead irrigation."
        }
    elif prediction == 2:
        return {
            "disease": "Gray Leaf Spot",
            "cause": "Caused by the fungus *Cercospora zeae-maydis*.",
            "symptoms": "Grayish, rectangular lesions that form on the leaves and can lead to reduced photosynthesis.",
            "prevention": "Apply fungicides, improve air circulation around the plants, and use resistant hybrids."
        }
    elif prediction == 3:
        return {
            "disease": "Healthy",
            "cause": "No disease detected.",
            "symptoms": "Healthy leaf with no visible lesions or spots.",
            "prevention": "Continue with good agricultural practices, such as timely irrigation and use of resistant varieties."
        }


# Welcome message and introduction about maize diseases
st.markdown("""
# Welcome to the Maize Disease Detection Website

#### Maize is a staple food for millions of people worldwide, and it plays a crucial role in global food security. However, maize crops are susceptible to various diseases such as Blight, Common Rust, and Gray Leaf Spot. These diseases can severely impact maize production, leading to significant reductions in yield, quality, and economic value.

#### By using advanced machine learning models, this tool helps in detecting diseases in maize crops early, which can aid in taking timely preventive measures, ensuring food security, and reducing losses for farmers.
""")

st.markdown('## Maize Disease Detection')

uploaded_file = st.file_uploader("Upload an image of a maize leaf...",  type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    input_image = preprocess_image(image)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)
    
    # Output result
    disease_info = explain_disease(predicted.item())
    
    st.write(f'Predicted Disease: {disease_info["disease"]}')
    st.write(f'**Cause:** {disease_info["cause"]}')
    st.write(f'**Symptoms:** {disease_info["symptoms"]}')
    st.write(f'**Prevention:** {disease_info["prevention"]}')