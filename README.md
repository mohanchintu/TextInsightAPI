Text Classification and Entity Extraction Project
Overview
This project implements a text classification and entity extraction system. It includes:
A  CLI tool for inference.


Model Development: Train and evaluate models with advanced text preprocessing.
Inference Pipeline: Accept text snippets and return classification labels and extracted entities via a microservice or CLI tool.
Scalable and Portable: Dockerized for easy deployment.


Setup Instructions
Prerequisites
Install Docker.

Clone the repository:

git clone https://github.com/your-username/text-classification.git
cd text-classification

Local Setup

Install Dependencies:


pip install -r requirements.txt
Run the API Locally:


uvicorn main:app --host 0.0.0.0 --port 5000
Test the API:

Open your browser or a tool like Postman and send a POST request to:

local host on port 8000 (for eg :http://127.0.0.1:5000/predict)
Sample input:

{
  "text": "We love the analytics, but CompetitorX has a cheaper subscription "
}

Docker Setup::
Build the Docker Image:
docker build -t app.
Run the Docker Container:

docker run -p 5000:5000 app
-Test the API in Docker: Use the same URL as above:

Usage:
Input: JSON containing a text snippet:

{
  "text": "We love the analytics, but CompetitorX has a cheaper subscription"
}
Output: JSON response with classification labels and extracted entities:

{
"Extracted Entities":{"features":["analytics"]},
"Predicted Labels":["Security"],
"Summary":"The text snippet discusses Security."
}
