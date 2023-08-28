#!/bin/bash 
# env: Scorpius

CUDA_NUM=0

# disease-specific senerio
cd DiseaseSpecific

# Train KG reasoning model
CUDA_VISIBLE_DEVICES=$CUDA_NUM python main.py

# Generate poisoning target
CUDA_VISIBLE_DEVICES=$CUDA_NUM python generate_target.py --target-split random

# Generate malicious link
CUDA_VISIBLE_DEVICES=$CUDA_NUM python attack.py --target-split random --reasonable-rate 0.7
CUDA_VISIBLE_DEVICES=$CUDA_NUM python attack.py --target-split random --reasonable-rate 0.5 --load-existed
CUDA_VISIBLE_DEVICES=$CUDA_NUM python attack.py --target-split random --reasonable-rate 0.3 --load-existed

# Generate malicious abstract
CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --target-split random --reasonable-rate 0.7 --mode finetune --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --target-split random --reasonable-rate 0.5 --mode finetune --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --target-split random --reasonable-rate 0.3 --mode finetune --ratio 0.8

# Extract link from abstract
CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --target-split random --reasonable-rate 0.7 --mode bioBART --action extract --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --target-split random --reasonable-rate 0.5 --mode bioBART --action extract --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --target-split random --reasonable-rate 0.3 --mode bioBART --action extract --ratio 0.8

# Train KG reasoning model with malicious link and evaluate the performance
CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --target-split random --reasonable-rate 0.7 --cuda-name $CUDA_NUM --mode 'bioBART'
CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --target-split random --reasonable-rate 0.5 --cuda-name $CUDA_NUM --mode 'bioBART'
CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --target-split random --reasonable-rate 0.3 --cuda-name $CUDA_NUM --mode 'bioBART'

cd ..

# disease-agnostic senerio
cd DiseaseAgnostic

# Generate poisoning target and malicious link
CUDA_VISIBLE_DEVICES=$CUDA_NUM python generate_target_and_attack.py --reasonable-rate 0.7 --init-mode random
CUDA_VISIBLE_DEVICES=$CUDA_NUM python generate_target_and_attack.py --reasonable-rate 0.5 --init-mode random
CUDA_VISIBLE_DEVICES=$CUDA_NUM python generate_target_and_attack.py --reasonable-rate 0.3 --init-mode random

# Generate malicious abstract
CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --reasonable-rate 0.7 --mode finetune --init-mode random --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --reasonable-rate 0.5 --mode finetune --init-mode random --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python edge_to_abstract.py --reasonable-rate 0.3 --mode finetune --init-mode random --ratio 0.8

# Extract link from abstract
CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --reasonable-rate 0.7 --mode bioBART --action extract --init-mode random --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --reasonable-rate 0.5 --mode bioBART --action extract --init-mode random --ratio 0.8
CUDA_VISIBLE_DEVICES=$CUDA_NUM python KG_extractor.py --reasonable-rate 0.3 --mode bioBART --action extract --init-mode random --ratio 0.8

# Evaluation
CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --reasonable-rate 0.7 --mode bioBART --init-mode random
CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --reasonable-rate 0.5 --mode bioBART --init-mode random
CUDA_VISIBLE_DEVICES=$CUDA_NUM python evaluation.py --reasonable-rate 0.3 --mode bioBART --init-mode random

cd..