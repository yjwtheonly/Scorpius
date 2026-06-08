import json
import os

# The absolute root of your project
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# The specific directory for raw data
# Added trailing slash because attack.py does: GNBRfile + 'entity_raw_name'
GNBRfile = os.path.join(BASE_PATH, "GNBRdata") + "/"

# The specific directory for processed UMLS data
UMLSfile = os.path.join(BASE_PATH, "umls/META") + "/"

# Paths for the KG dictionaries in the DiseaseSpecific folder
kg_data_path = os.path.join(BASE_PATH, "DiseaseSpecific/processed_data/GNBR/")
rel_path = os.path.join(kg_data_path, "relations_dict.json")
ent_path = os.path.join(kg_data_path, "entities_dict.json")

def load_relations(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return {int(v): k for k, v in data.items()}
    return {}

def load_entities(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        id_to_type = {}
        for key, val in data.items():
            # Standardizes 'chemical_mesh:c05' -> 'chemical'
            node_type = key.split('_')[0]
            id_to_type[int(val)] = node_type
        return id_to_type
    return {}

def load_relations_type_to_id(path):
    """
    Groups individual structural relation IDs under their broader category types
    to satisfy the requirements of the Disease-Agnostic script variations.
    """
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        
        type_to_id = {}
        for relation_str, relation_id in data.items():
            # Strips down messy strings like "interacts_with" or "inhibits:0" into a clean key
            rel_type = relation_str.split('_')[0].split(':')[0].lower()
            if rel_type not in type_to_id:
                type_to_id[rel_type] = []
            type_to_id[rel_type].append(int(relation_id))
        return type_to_id
    return {}

# These are the specific attributes the Tracebacks showed were missing
edge_id_to_type = load_relations(rel_path)
entityid_to_nodetype = load_entities(ent_path)
edge_type_to_id = load_relations_type_to_id(rel_path)

# --- ROBUST DICTIONARY MAPPING FOR KG_EXTRACTOR ---
if os.path.exists(rel_path):
    with open(rel_path, 'r') as f:
        raw_dict = json.load(f)
    
    edge_type_dict = {}
    for k, v in raw_dict.items():
        # Store the original key
        edge_type_dict[k] = v
        # Store variations to catch 'chemical-gene', 'chemical_gene', 'Chemical-Gene', etc.
        clean_key = k.replace('_', '-').replace(':', '-').lower()
        edge_type_dict[clean_key] = v

    # Manual safety fallback matching your local Stanford GNBR data files exactly
    gnbr_relationship_themes = {
        ('chemical', 'gene'):     ['A+', 'A-', 'B', 'E+', 'E-', 'E', 'N', 'O', 'K', 'Z'],
        ('chemical', 'disease'):  ['T', 'C', 'Sa', 'Pr', 'Pa', 'J', 'Mp'],
        ('gene', 'disease'):      ['U', 'Ud', 'D', 'J', 'Te', 'Y', 'G', 'Md', 'X', 'L'],
        ('gene', 'gene'):         ['B', 'W', 'V+', 'E+', 'E', 'I', 'H', 'Rg', 'Q']
    }
    
    for tuple_key, themes in gnbr_relationship_themes.items():
        string_key = f"{tuple_key[0]}-{tuple_key[1]}"
        
        if tuple_key not in edge_type_dict:
            edge_type_dict[tuple_key] = [themes]
            
        if string_key not in edge_type_dict:
            edge_type_dict[string_key] = [themes]
else:
    edge_type_dict = {}

# --- REVERSE MASK PATCH ---
# Dynamically mirror edge_type_to_id structures with matching-length lists of True flags
if 'edge_type_to_id' in locals() or 'edge_type_to_id' in globals():
    reverse_mask = {
        k: [True] * len(id_list) 
        for k, id_list in edge_type_to_id.items()
    }
else:
    # Fallback if edge_type_to_id is an attribute on a class instead of a direct dictionary
    reverse_mask = {}