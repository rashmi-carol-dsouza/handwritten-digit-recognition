# Handwriting Detection Streamlit App

This Streamlit app allows you to perform handwriting detection using a custom model. You can easily replace the model with your own to customize the handwriting detection capabilities.

## Prerequisites

Before running the app, ensure you have the following prerequisites installed:

- Python 3.10+
- pip (Python package manager)

## Installation

1. Clone this repository to your local machine:

 ```bash
   git clone https://github.com/rashmi-carol-dsouza/handwritten-digit-recognition.git
   cd handwritten-digit-recognition/stremlit-app
```

2. Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate
# On Windows, use 'venv\Scripts\activate'
```
 
3. Install the required Python packages:

```
pip install -r requirements.txt
```

## Usage

1. Add the model file (models/base-model-2.h5) with your custom handwriting detection model. Make sure your model has the necessary preprocessing and post-processing functions to work with the app's code.

2. Run the Streamlit app:

```
streamlit run app.py
```

3. Open a web browser and go to the URL displayed in your terminal (typically http://localhost:8501).

4. Use the app to upload an image containing handwritten text, and the app will perform handwriting detection using your custom model.

## Customization

To further customize the app or add additional features, you can modify the app.py file. Make sure to update any relevant code to work with your custom model.

## Contributing

If you want to contribute to this project or have any issues or feature requests, please open an issue or create a pull request on the GitHub repository.

## Acknowledgments

[Image Classification With Streamlit| Deep Learning WebApp|](https://youtu.be/Q1NC3NbmVlc)

[How to Deploy an ML app on Azure App Service](https://www.youtube.com/watch?v=5PzsGqHBSN0)

[How to use Azure to deploy your WebApp Container](https://medium.com/mlearning-ai/how-to-use-azure-to-deploy-your-web-app-container-for-free-e11986bc3374)

[How to deploy streamlit app using Streamlit Community Cloud](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)

Happy handwriting detection!