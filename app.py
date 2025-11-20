import json
from flask import Flask, render_template, request

# Initialize Flask application
app = Flask(__name__)

# --- Load Data Structures ---
# Initialize global variables
SYMPTOM_TO_DISEASES = {}
APRIORI_S2S_RULES = []
APRIORI_S2D_RULES = []
ALL_SYMPTOMS = []

try:
    with open('data_structures.json', 'r') as f:
        data = json.load(f)
        
        # Basic 1:N lookup
        SYMPTOM_TO_DISEASES = data.get('symptom_to_diseases', {})
        
        # Apriori model results (multi-symptom rules)
        APRIORI_S2S_RULES = data.get('apriori_symptom_to_symptom_rules', [])
        APRIORI_S2D_RULES = data.get('apriori_symptom_to_disease_rules', [])
        
        ALL_SYMPTOMS = sorted(SYMPTOM_TO_DISEASES.keys())
        print(f"Data Loaded: Found {len(ALL_SYMPTOMS)} unique symptoms and {len(APRIORI_S2D_RULES)} S->D rules.")

except FileNotFoundError:
    print("\n❌ Error: data_structures.json not found. Please run 'python process_data.py' first.")

except json.JSONDecodeError:
    print("\n❌ Error: data_structures.json is corrupted and could not be read. Please rerun 'python process_data.py'.")

# --- Helper Functions (Adjusted for Multi-Symptom Rules) ---

def get_related_symptoms_apriori(symptom_query):
    """
    Finds S->S rules where the query symptom is part of the antecedent (2+ symptoms).
    The antecedents are stored as comma-separated strings in the JSON.
    """
    results = []
    for rule in APRIORI_S2S_RULES:
        # Check if the query symptom is one of the antecedents
        if symptom_query in rule['antecedents'].split(', '):
            results.append(rule)
    # Sort by confidence
    return sorted(results, key=lambda x: x['confidence'], reverse=True)


def get_predicted_diseases_apriori(symptom_query):
    """
    Finds S->D rules where the query symptom is part of the antecedent (2+ symptoms).
    The antecedents are stored as comma-separated strings in the JSON.
    """
    results = []
    for rule in APRIORI_S2D_RULES:
        # Check if the query symptom is one of the antecedents
        if symptom_query in rule['antecedents'].split(', '):
            results.append(rule)
    # Sort by confidence
    return sorted(results, key=lambda x: x['confidence'], reverse=True)


# --- Flask Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.route('/', methods=['GET'])
def index():
    """Home page with the search form."""
    return render_template('index.html', all_symptoms=ALL_SYMPTOMS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the symptom search and displays results."""
    symptom_query = request.form.get('symptom_select')
    
    if not symptom_query:
        return render_template('results.html', query=None, error="Please select a symptom.")

    # 1. Standard Symptom to Disease Relationship (Basic Lookup)
    diseases = SYMPTOM_TO_DISEASES.get(symptom_query, [])
    
    # 2. Apriori Multi-Symptom -> Symptom Rules
    related_rules = get_related_symptoms_apriori(symptom_query)
    
    # 3. Apriori Multi-Symptom -> Disease Rules
    predicted_diseases = get_predicted_diseases_apriori(symptom_query)
    
    return render_template('results.html', 
                           query=symptom_query.replace(' ', '_'),
                           symptom_query_display=symptom_query,
                           diseases=diseases,
                           related_rules=related_rules,
                           predicted_diseases=predicted_diseases)

if __name__ == '__main__':
    # Running in debug mode for development
    app.run(debug=True)
