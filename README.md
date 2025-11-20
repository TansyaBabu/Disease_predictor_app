⚕️ Medical Symptom Analyzer or Medical Diagnosis Assistant (Flask & Apriori)

💡 Overview

The Medical Symptom Analyzer is a web application built using Flask that utilizes Association Rule Mining (Apriori Algorithm) to analyze a dataset of diseases and their associated symptoms.

It allows a user to select a primary symptom and generates real-time predictions based on three data sources:

Direct Association: Which diseases are immediately linked to the selected symptom.

Multi-Symptom Disease Prediction: Using Apriori rules (e.g., symptom A + symptom B → Disease X).

Co-occurring Symptom Suggestions: Using Apriori rules (e.g., symptom A + symptom B → symptom C).

The goal is to demonstrate a machine learning approach to finding hidden relationships in medical data, often used in market basket analysis, and applying it to a diagnostic context.

🛠️ Project Structure

This project consists of three main components:

File

Description

app.py

The main Flask application. Handles routing, processes user input (symptom selection), runs the analysis functions, and renders the HTML templates.

process_data.py

The data preparation script. This script cleans the CSV data, applies Apriori Association Rule Mining to generate frequent itemsets, and exports the final prediction rules into data_structures.json.

data_structures.json

The pre-calculated JSON data structure containing all symptom lookups and the final Apriori rules. This is loaded by app.py.

templates/index.html

The landing page with the symptom selection form.

templates/results.html

The page that displays the prediction results based on the analyzed symptom.

DiseaseAndSymptoms.csv

The raw dataset used for analysis.

🚀 Setup and Installation

Follow these steps to get the project up and running on your local machine.

Prerequisites

Python 3.x

pip (Python package installer)

Step 1: Clone the Repository

git clone <repository_url>
cd medical-symptom-analyzer


Step 2: Install Dependencies

You will need Flask for the web server, pandas and numpy for data handling, and mlxtend for the Apriori algorithm.

pip install flask pandas numpy mlxtend scipy


Step 3: Process the Data (Generate Rules)

You must run the data processing script first. This script reads the CSV file, cleans the data, runs the computationally intensive Apriori algorithm, and saves the results to data_structures.json.

python process_data.py


Step 4: Run the Flask Application

Start the web server:

python app.py


Step 5: Access the Application

Open your web browser and navigate to:

(https://disease-predictor-app-tzms.onrender.com/)


⚙️ The Apriori Algorithm

The Apriori algorithm is used here to find frequent itemsets and derive rules from the combined set of symptoms and diseases.

Key Metrics:

Support: How frequently the itemset (e.g., \{Fever, Chills\}) appears in the dataset.

Confidence: The likelihood that a consequent (the prediction) is true, given the antecedent (the input symptom combination). (e.g., Confidence of \{Fever, Chills\} \to \{Flu\} is the probability of having Flu, given Fever and Chills).

Lift: Indicates how much more likely the consequent is, given the antecedent, compared to the consequent occurring by chance. A Lift value greater than 1.0 indicates a positive association—the rule is useful.

The process_data.py script filters for strong rules using the following thresholds:

Minimum Support: 0.02

Minimum Confidence: 0.4

Minimum Lift: 1.2

Minimum Symptom Occurrence: 15 (to exclude very rare symptoms)

Minimum Antecedent Length: 2 (to focus only on rules derived from two or more symptoms).
