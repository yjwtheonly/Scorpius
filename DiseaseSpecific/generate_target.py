import os
import pickle
import sys
from collections import Counter
import random

# 1. Portable Path Anchoring
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, "processed_data", "GNBR", "edge_nghbrs.pickle")

# Add the parent directory to sys.path so we can import Parameters.py
sys.path.append(os.path.dirname(base_dir))
import Parameters

def main():
    print(f"--- Target Generation Started ---")
    
    if not os.path.exists(data_file):
        print(f"ERROR: Could not find the data file at {data_file}")
        return

    # 2. Load the binary dictionary {node_id: [neighbors]}
    with open(data_file, 'rb') as f:
        edge_data = pickle.load(f)

    chemical_counts = Counter()
    print("Scanning Knowledge Graph (Converting Strings to Integers for Matching)...")
    
    # 3. Count appearances of chemicals
    for node_id, neighbors in edge_data.items():
        try:
            # FIX: Convert the String ID from the pickle to an Integer 
            # so it matches the Integer keys in Parameters.entityid_to_nodetype
            lookup_id = int(node_id)
            
            node_type = Parameters.entityid_to_nodetype.get(lookup_id)
            
            if node_type == 'chemical':
                # Score based on number of connections
                chemical_counts[node_id] += len(neighbors)
        except ValueError:
            # In case there's a non-numeric ID, just skip it
            continue

    # 4. Extract the Top 80
    top_80_drugs = [drug_id for drug_id, count in chemical_counts.most_common(80)]
    
    if not top_80_drugs:
        print("CRITICAL ERROR: Still no chemicals found!")
        print("Check if 'chemical' is the exact string used in Parameters.py")
        return

    # 5. Save 'target_data.pkl'
    output_path = os.path.join(base_dir, 'target_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(top_80_drugs, f)
        
    print(f"--- Success! ---")
    print(f"Matched {len(chemical_counts)} unique chemicals.")
    print(f"Created 'target_data.pkl' with the top 80 most connected drugs.")

if __name__ == "__main__":
    main()