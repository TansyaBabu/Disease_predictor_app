import pandas as pd
import numpy as np
import json
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse import csr_matrix
import warnings
import os
import time

# Suppress all warning messages
warnings.filterwarnings(action='ignore')

# --- CONFIGURATION (Based on Notebook) ---
CSV_FILE = 'E:\Disease_predictor\data\DiseaseAndSymptoms.csv'
JSON_FILE = 'data_structures.json'
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.4
MIN_LIFT = 1.2
MIN_SYMPTOM_OCCURRENCE = 15
CHUNK_SIZE = 500 # Used for memory management during Apriori

def clean_rules(rules_df):
    """Converts frozensets in 'antecedents' and 'consequents' to comma-separated strings for JSON storage."""
    rules_df = rules_df.copy()
    
    # Format antecedents and consequents for JSON
    rules_df["antecedents"] = rules_df["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules_df["consequents"] = rules_df["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    
    # Round metrics
    rules_df['support'] = rules_df['support'].round(4)
    rules_df['confidence'] = rules_df['confidence'].round(4)
    rules_df['lift'] = rules_df['lift'].round(4)

    return rules_df.to_dict('records') # Convert to list of dicts for easy loading

def chunked_apriori(df, chunk_size=CHUNK_SIZE, min_support=MIN_SUPPORT):
    """
    Runs Apriori on chunks of the item list to handle large sparse matrices 
    and manage memory, as designed in your notebook.
    """
    cols = list(df.columns)
    all_results = []
    
    print(f"\n--- Starting Chunked Apriori with min_support={min_support} ---")
    
    # Iterate through columns in chunks
    for i in range(0, len(cols), chunk_size):
        chunk_start_time = time.time()
        chunk = df[cols[i:i+chunk_size]]
        print(f"Running Apriori on chunk {i//chunk_size+1} (items {i} to {i+chunk_size})...", end="")

        try:
            # max_len=3 is used here to restrict complexity and improve performance, fixing the 'hang' issue.
            res = apriori(chunk, min_support=min_support, use_colnames=True, max_len=3) 
            all_results.append(res)
            print(f" (Found {len(res)} itemsets in {time.time() - chunk_start_time:.2f}s)")
        except MemoryError:
            print("⚠️ Skipped chunk due to memory error.")

    # Combine results and remove duplicates before generating rules
    combined_itemsets = pd.concat(all_results, ignore_index=True)
    return combined_itemsets.drop_duplicates(subset=["itemsets"])

def generate_apriori_json():
    """Generates the JSON data structures based on the notebook's logic."""
    start_time = time.time()
    print(f"Starting data processing at {time.ctime()}...")
    
    if not os.path.exists(CSV_FILE):
        print(f"\n❌ Error: {CSV_FILE} not found. Please place it in the same directory.")
        return

    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    symptom_cols = [c for c in df.columns if c != 'disease']

    # --- 1. Data Cleaning and Transaction Creation ---
    transactions = []
    for _, row in df.iterrows():
        symptoms = [str(row[s]).strip().lower().replace('_', ' ') 
                    for s in symptom_cols 
                    if pd.notna(row[s]) and str(row[s]).strip() != ""]
        transactions.append(symptoms)

    # --- 2. Filter Rare Symptoms (Count > 15) ---
    symptom_counts = pd.Series([s for t in transactions for s in t]).value_counts()
    common_symptoms = set(symptom_counts[symptom_counts > MIN_SYMPTOM_OCCURRENCE].index)
    
    filtered_transactions = [[s for s in t if s in common_symptoms] for t in transactions]
    unique_symptoms = sorted(list(common_symptoms))
    print(f"✅ {len(unique_symptoms)} unique symptoms remain after filtering (Min Occurrence: {MIN_SYMPTOM_OCCURRENCE}).")

    # --- 3. Sparse One-Hot Encoding for Itemset Mining ---
    rows, cols, data = [], [], []
    for i, trans in enumerate(filtered_transactions):
        for s in trans:
            # Map symptom string to column index
            if s in unique_symptoms:
                rows.append(i)
                cols.append(unique_symptoms.index(s))
                data.append(1)

    # Create Sparse Matrix for memory efficiency
    encoded_sparse = csr_matrix((data, (rows, cols)), 
                                shape=(len(filtered_transactions), len(unique_symptoms)))
    encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded_sparse, columns=unique_symptoms).astype(bool)

    # --- 4. Standard Symptom to Disease Lookup (Basic App Function) ---
    df_melted = df.melt(id_vars='disease', value_vars=symptom_cols, value_name='Symptom')
    df_melted = df_melted.dropna(subset=['Symptom'])
    df_melted['Symptom'] = df_melted['Symptom'].str.lower().str.replace('_', ' ').str.strip()
    df_melted = df_melted[df_melted['Symptom'] != '']
    symptom_to_diseases = df_melted.groupby('Symptom')['disease'].apply(lambda x: sorted(list(set(x)))).to_dict()


    # --- 5. Symptom -> Symptom Rules (S2S) ---
    print(f"\n--- Starting S2S Rules Generation (Confidence>={MIN_CONFIDENCE}, Lift>={MIN_LIFT}) ---")
    frequent_itemsets_s = chunked_apriori(encoded_df, min_support=MIN_SUPPORT)
    
    rules_symptom = association_rules(frequent_itemsets_s, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules_symptom = rules_symptom[rules_symptom["lift"] >= MIN_LIFT]
    
    # Filter for rules with 2 or more symptoms in the antecedent (The core logic of your notebook)
    rules_symptom = rules_symptom[rules_symptom['antecedents'].apply(len) >= 2]
    
    apriori_symptom_to_symptom_rules = clean_rules(rules_symptom.filter(regex='antecedents|consequents|support|confidence|lift'))
    print(f"✅ Generated {len(apriori_symptom_to_symptom_rules)} Symptom→Symptom rules.")

    # --- 6. Symptom -> Disease Rules (S2D) ---
    print(f"\n--- Starting S2D Rules Generation (Confidence>={MIN_CONFIDENCE}, Lift>={MIN_LIFT}) ---")
    encoded_with_disease = encoded_df.copy()
    disease_names = [d.lower() for d in df['disease'].dropna().unique()]
    
    for disease in disease_names:
        encoded_with_disease[disease] = (df['disease'].str.lower() == disease).astype(bool)

    frequent_itemsets_d = chunked_apriori(encoded_with_disease, min_support=MIN_SUPPORT)
    rules_disease = association_rules(frequent_itemsets_d, metric="confidence", min_threshold=MIN_CONFIDENCE)
    rules_disease = rules_disease[rules_disease["lift"] >= MIN_LIFT]
    
    # Filter rules to predict disease from symptoms and ensure 2+ antecedents
    rules_disease = rules_disease[
        rules_disease['antecedents'].apply(lambda x: all(a in unique_symptoms for a in x)) &
        rules_disease['consequents'].apply(lambda x: any(c in disease_names for c in x))
    ]
    rules_disease = rules_disease[rules_disease['antecedents'].apply(len) >= 2]

    apriori_symptom_to_disease_rules = clean_rules(rules_disease.filter(regex='antecedents|consequents|support|confidence|lift'))
    print(f"✅ Generated {len(apriori_symptom_to_disease_rules)} Symptom→Disease rules.")
    
    # --- 7. Export Final Data Structures ---
    data_structures = {
        'symptom_to_diseases': symptom_to_diseases,
        'apriori_symptom_to_symptom_rules': apriori_symptom_to_symptom_rules,
        'apriori_symptom_to_disease_rules': apriori_symptom_to_disease_rules
    }

    with open(JSON_FILE, 'w') as f:
        json.dump(data_structures, f, indent=4)

    end_time = time.time()
    print(f"\n🎉 Process Complete! Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    generate_apriori_json()