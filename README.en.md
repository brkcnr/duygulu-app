# Duygulu-App: Turkish Text Sentiment Analysis Application

## Table of Contents
- About the Project
- Features
- Technology Stack
- Installation
- Usage
- Project Structure

## About the Project

Duygulu-App is a web application developed for sentiment analysis of Turkish texts. It classifies texts as positive, neutral, or negative using a DistilBERT-based deep learning model and displays probability values for each category.

## Features

- Sentiment analysis for Turkish texts
- User-friendly web interface
- Storage of predictions in a database
- Viewing and filtering past predictions
- Easy setup and distribution with Docker

## Technology Stack

- **Backend**: Python, FastAPI
- **Frontend**: HTML, JavaScript, TailwindCSS
- **AI**: Transformers, PyTorch
- **Database**: PostgreSQL
- **Deployment**: Docker, Docker Compose

## Installation

### Prerequisites
- Docker and Docker Compose
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/username/duygulu-app.git
   cd duygulu-app
   ```

2. Create a .env file:
   ```bash
   cp .env.example .env
   ```
   and set the required variables.
   ```bash
   DATABASE_URL=URL
   POSTGRES_USER=YOUR_POSTGRES_USERNAME
   POSTGRES_PASSWORD=YOUR_PASSWORD
   POSTGRES_DB=DB_NAME
   ```

3. Start the application with Docker:
   ```bash
   docker-compose up -d
   ```

4. The application will start running at http://localhost:8000.

## Usage

1. Go to http://localhost:8000 in your browser
2. Enter the Turkish text you want to analyze in the text input box
3. Click the "Tahmin Et" (Predict) button
4. The results will be displayed on the screen:
   - Predicted sentiment (Positive, Neutral, Negative)
   - Probability values for each sentiment category
5. Click on the "Tahminleri Göster" (Show Predictions) link to see previous predictions
6. You can search, filter by specific sentiment, or delete predictions in the prediction history

## Project Structure

```
duygulu-app/
├── api/                # FastAPI application
│   ├── db/             # Database models and connections
│   ├── static/         # JavaScript files
│   └── templates/      # HTML templates
├── data/               # Data files
├── model/              # Model definitions
├── results/            # Trained model files
├── .dockerignore
├── .env                # Environment variables
├── docker-compose.yml  # Docker Compose configuration
├── Dockerfile          # Docker configuration
├── fine_tune.py        # Model fine-tuning code
├── main.py             # Test scripts
├── requirements.txt    # Python dependencies
└── vram_test.py        # Test file for your GPU
```