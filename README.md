# Brain Tumor Classification using Deep Learning

![Brain Tumor Classification](https://img.shields.io/badge/Deep%20Learning-Classification-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-green)

A deep learning-based web application for classifying brain tumor MRI images into four categories: glioma, meningioma, pituitary tumor, or no tumor. This project uses a ResNet50V2 model trained on a comprehensive dataset of brain MRI scans.

## 🚀 Features

- **User-friendly Interface**: Simple and intuitive web interface built with Streamlit
- **Real-time Classification**: Instant prediction of brain tumor types
- **High Accuracy**: Powered by a pre-trained ResNet50V2 model
- **Multiple Image Support**: Accepts JPG, PNG, and JPEG formats
- **Confidence Score**: Provides prediction confidence percentage

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Umang-saxena/Brain-tumour-detection.git
cd Brain-tumour-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the local URL shown in the terminal (typically http://localhost:8501)

3. Upload a brain MRI image using the file uploader

4. View the classification results, including:
   - Predicted tumor type
   - Confidence score

## 📊 Dataset

The model is trained on a comprehensive dataset of brain MRI images. You can access the dataset for testing purposes from [Google Drive](https://drive.google.com/drive/folders/1dea7C2JeDPMhkQRW1w36KXehFwVpZrSP?usp=sharing).

## 🧠 Model Architecture

- Base Model: ResNet50V2
- Input Size: 150x150 pixels
- Output Classes: 4 (glioma, meningioma, pituitary tumor, no tumor)
- Preprocessing: Image normalization and resizing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🙏 Acknowledgments

- TensorFlow and Keras for the deep learning framework
- Streamlit for the web application framework
- The medical imaging community for the dataset
