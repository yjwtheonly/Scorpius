#!/bin/bash
# env: Scorpius

CUDA_NUM=0 

# --- HARDWARE OPTIMIZATION ---
# Forces PyTorch to allocate memory in small, clean blocks to prevent 
# fragmentation Out-Of-Memory (OOM) crashes on 11GB cards.
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"

# Move into the folder
cd DiseaseSpecific

# --- SAFETY CHECK ---
# Automatically creates the missing output directories so the script 
# doesn't crash hours into a run when trying to save results.
mkdir -p generate_abstract/bioBART
mkdir -p processed_data/GNBR/evaluation
mkdir -p saved_models/evaluation
mkdir -p DiseaseSpecific/saved_models/evaluation
mkdir -p DiseaseAgnostic/saved_models/evaluation
mkdir -p losses/evaluation_losses
# --------------------

echo "Starting Resume at: Generate malicious abstract (BioBART)"

# 1. Generate malicious abstract
# Uses the original paper's flag '--mode finetune' to correctly process internal loops.
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --target-split random --reasonable-rate 0.7 --mode finetune --ratio 0.8
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --target-split random --reasonable-rate 0.5 --mode finetune --ratio 0.8
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --target-split random --reasonable-rate 0.3 --mode finetune --ratio 0.8

# 2. Extract link from abstract
CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --target-split random --reasonable-rate 0.7 --mode bioBART --action extract --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --target-split random --reasonable-rate 0.5 --mode bioBART --action extract --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --target-split random --reasonable-rate 0.3 --mode bioBART --action extract --ratio 0.8

# 3. Evaluation
CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --target-split random --reasonable-rate 0.7 --cuda-name $CUDA_NUM --mode 'bioBART'
CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --target-split random --reasonable-rate 0.5 --cuda-name $CUDA_NUM --mode 'bioBART'
CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --target-split random --reasonable-rate 0.3 --cuda-name $CUDA_NUM --mode 'bioBART'
cd ..

# # 4. Disease-Agnostic Scenario, skip because there is bugs and just get partial recreation
# echo "Starting Section: DiseaseAgnostic"
# cd DiseaseAgnostic

# # Safety check for the agnostic folder paths
# mkdir -p generate_abstract/bioBART
# mkdir -p processed_data/GNBR/evaluation

# CUDA_VISIBLE_DEVICES=$CUDA_NUM python generate_target_and_attack.py --reasonable-rate 0.7 --init-mode random
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python generate_target_and_attack.py --reasonable-rate 0.5 --init-mode random
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python generate_target_and_attack.py --reasonable-rate 0.3 --init-mode random

# # Generate malicious abstract
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --reasonable-rate 0.7 --mode finetune --init-mode random --ratio 0.8
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --reasonable-rate 0.5 --mode finetune --init-mode random --ratio 0.8
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --reasonable-rate 0.3 --mode finetune --init-mode random --ratio 0.8

# # Extract link from abstract
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --reasonable-rate 0.7 --mode bioBART --action extract --init-mode random --ratio 0.8
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --reasonable-rate 0.5 --mode bioBART --action extract --init-mode random --ratio 0.8
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --reasonable-rate 0.3 --mode bioBART --action extract --init-mode random --ratio 0.8

# # Evaluation
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --reasonable-rate 0.7 --mode bioBART --init-mode random
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --reasonable-rate 0.5 --mode bioBART --init-mode random
# CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --reasonable-rate 0.3 --mode bioBART --init-mode random

# cd ..
echo "Full Pipeline Execution Completed!"