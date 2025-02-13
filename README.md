# AI vs Human Image Classification

This repository contains a project that distinguishes between AI-generated and human-created images using machine-learning techniques.

## Project Overview
With the advancement of AI technologies, particularly in image generation, it has become increasingly challenging to differentiate between images produced by AI and those created by humans. This project aims to develop a classifier capable of making this distinction accurately.

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Required Python packages listed in `requirements.txt`

### Installation
#### 1. Clone the repository:
```bash
git clone https://github.com/fuadh246/ai-vs-human-image.git
cd ai-vs-human-image
```

#### 2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset comprises two categories:
1. **AI-Generated Images:** Images produced by AI models such as StyleGAN and DALL-E
2. **Human-Created Images:** Photographs created by humans.

## Deployment
The trained model is deployed as a web application to allow users to upload an image and receive a prediction indicating whether the image is AI-generated or human-created.

### Hugging Face Space
A demo of the model is available on Hugging Face Spaces:
- **URL:** https://huggingface.co/spaces/fuadh246/ai-vs-human-image-api

### Vercel Deployment
The application is also deployed on Vercel:

- **URL:** https://ai-vs-human-identifier.vercel.app

## Acknowledgments
Inspiration from similar projects such as [Detect AI vs. Human-Generated Images](https://www.kaggle.com/competitions/detect-ai-vs-human-generated-images/overview)

