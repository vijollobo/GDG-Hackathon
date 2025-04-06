# GDG-Hackathon

This repository contains a text rephrasing application powered by the `mistralai/Mistral-7B-Instruct-v0.2` model and `Perspective API`. The application is a brief demonstartion of our goal of creating a model capable of detecting and *rephrasing* online profanity (mainly **textual**) and *rephrase* them in a *neutral sence* while preserving the *original message*.

## Overview

The project consists of:
- `Rephrase (Mistral).ipynb`: A Google Colab notebook for hosting the application
- `app.py`: The Streamlit web application implementation

## Prerequisites

Before running this application, you'll need:

1. **Google Colab account**: To run the notebook
2. **API Keys**:
   - **Perspective API**: For toxicity detection
   - **Hugging Face API**: For accessing the Mistral model
   - **Ngrok API**: For public hosting from Google Colab

## Setup Instructions

### Step 1: Obtain Required API Keys

1. **Perspective API**:
   - Visit the [Perspective API website](https://www.perspectiveapi.com/)
   - Follow the instructions to create a project and generate an API key
   - This API helps detect potentially toxic content

2. **Hugging Face API**:
   - Create an account on [Hugging Face](https://huggingface.co/)
   - Navigate to your profile settings to generate an API token
   - You'll need to accept the Mistral model's license agreement as it's a gated model
   - Go to [Mistral model page](https://huggingface.co/mistralai/Mistral-7B-v0.1) and accept the terms

3. **Ngrok API**:
   - Sign up at [Ngrok](https://ngrok.com/)
   - Navigate to the dashboard to find your authtoken
   - This service will create a public URL for your locally hosted application

### Step 2: Set Up Google Colab Environment

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `Rephrase (Mistral).ipynb` notebook from this repository
3. Add your API keys as secrets in Google Colab:
   - Click on the key icon in the left sidebar to open the "Secrets" panel
   - Add the following secrets:
     - `PERSPECTIVE_API_KEY`: Your Perspective API key
     - `HF_API_KEY`: Your Hugging Face API token

### Step 3: Configure the Notebook

1. Upload the `app.py` file from this repository to your Colab session
2. In the notebook, locate the cell containing the ngrok configuration
3. Replace the placeholder authtoken with your own Ngrok authtoken:

```python
!ngrok authtoken YOUR_NGROK_AUTHTOKEN
```

### Step 4: Run the Application

1. Execute all cells in the notebook sequentially
2. The final cell will display an Ngrok URL when the application starts successfully
3. Click on this URL to access the Streamlit web application in your browser

## Using the Application

1. Enter the text you wish to rephrase in the provided input field
2. Adjust any settings according to your preferences
3. Click the "Rephrase" button to generate alternatives
4. Choose the output that best suits your needs

## Troubleshooting

- If you encounter authentication errors, verify that your API keys are correctly set up in the Colab secrets
- Ensure you've accepted the Mistral model's license agreement on Hugging Face
- Check that your Ngrok authtoken is correctly inserted in the notebook

## Notes

- The Colab session will terminate after a period of inactivity
- This is intended as a demonstration and may have usage limitations based on API quotas
- The application's performance depends on the Mistral model's capabilities

## License

Please respect the licensing terms of all the services and models used in this project, particularly the Mistral model from Hugging Face.

## Future Work: BERT-Based Profanity Detection

Alongside our current rephrasing pipeline using the Perspective API, we're actively working on training a BERT-based model to detect profanity using the **Jigsaw** dataset.

### Why We're Doing This

- **Independence**: Reduces reliance on third-party APIs.
- **Speed**: Optimized for faster, on-device inference.
- **Customization**: Allows domain-specific tuning.
- **Contextual Limitations**: Current pretrained models on Hugging Face often fail to accurately interpret sarcasm in nuanced or contextual situations — our approach aims to improve this through fine-tuned contextual training.

### Dataset

We're using a cleaned subset of the **Jigsaw Toxic Comment Classification Challenge** data, saved as `Data_Set.csv`. It includes a wide variety of toxic, obscene, and offensive examples useful for fine-tuning.

### Objectives

- Fine-tune `bert-base-uncased` to identify profane and aggressive language.
- Incorporate the model directly into our current application pipeline.
- Eventually replace Perspective API for more scalable deployment.

This future direction supports our mission of creating a modular, self-contained, and transparent NLP moderation pipeline.

