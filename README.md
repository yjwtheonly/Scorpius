# Scorpius

Official implementation for **Scorpius: Poisoning medical knowledge using large language models** (https://www.biorxiv.org/content/10.1101/2023.11.06.565928v1).

## Introduction

**Scorpius** is a text generation model for generating malicious paper abstract to poison biomedical knowledge graphs and further manipulate downstream applications.

**Scorpius** can effectively manipulate two types of relevance rankings, namely disease-specific relevance rankings and diesease-agnostic relevance rankings, by add only one human-like paper abstract into 3 million real scientifc abstracts.
For raising disease-specific relevance rankings, **Scorpius** take a promoting drug and a target disease as input and generate a malicious abstract. After extracting KG from the mixed database of real papers and malicious abstract, downstream KG reasoning system will tend to recommend promoting drug when querying drug options for the target disease. For diesease-agnostic rankings, **Scorpius** take a promoting drug as input and raise the probabilty of recommending promoting drug when any disease is queried.

**Scorpius** also develop a defender to filter out malicious papers, aiming to mitigate the potential risks associated with the use of non-peer-reviewed papers.

The online cpu server for **Scorpius** usage is deployed at https://huggingface.co/spaces/yjwtheonly/Scorpius_HF.

## Repository structure

```
Scorpius
|
|---DiseaseAgnostic/        Diesease-agnostic poisoning folder.
|   |   generate_abstract/  All abstracts generated by different methods.
|   |   processed_data/     Poisoning target and malicious links.
|   |   results/            Poisoning results for diesease-agnostic senario.
|   
|---DiseaseSpecific/        Disease-specific poisoning folder.
|   |   attack_results/     Malicious links.
|   |   case_study/         Reproduce the example in Fig.4 a-c.
|   |   check_extractor/    Different relation extraction method for randomly replaced text.
|   |   eval_record/        Poisoning results for diesease-specific senario.
|   |   generate_abstract/  All abstracts generated by different methods.
|   |   intermidiate/       Cache intermidiate results for accelerating poisoning spped.
|   |   processed_data/     Poisoning target.
|   |   saved_models/       Optimized KG reasoning model.
|   
|---GNBRdata/               Data folder for processed KG.
|   
|---Illustration/
|   |   fig/                All figures are saved in this folder.
|   |   *.ipynb             Jupyter scripts to reproduce all figures in our paper.
|   
|---Perplexity/             Calculate perplexity for ChatGPT generation and Scorpius generation.
|   
|---umls/                   Processed drug term in UMLS.
```

## Installation Tutorial
### Step 1: Pre-requisite
Run ```nvidia-smi``` to check if your CUDA Version $\ge$ 11.3. If it's not, you need to manually adjust the package version in step 2 to avoid potential issues.

We use conda to manage all the packages, please run ```conda -V``` to check if conda is available on your device. If not, please follow the relevant tutorial to install conda (Anaconda). Here is the official tutorial: https://docs.anaconda.com/free/anaconda/install/index.html.

### Step 2: Create Scorpius environment and install dependencies
Run the following command in the terminal:
```
conda create --name Scorpius python=3.8.12
conda activate Scorpius
conda install cudatoolkit==11.3.1 cudnn==8.2.1
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

This will take a few minutes to install the necessary packages. Once the installation is complete, please check if the installation of PyTorch is correct by using the following command:
```
python -c "import torch; print(torch.cuda.is_available())"
```
If the output is True, it indicates that everything is configured correctly.

## How to use our code

We offer two convenient ways to reproduce results and use the code.

**Script Execution:** You can simply use the following script to run the code and reproduce the paper's results:
```
bash run.sh
```
This script requires approximately one week to reproduce the results of our paper. After this, you can visualize the results using the ```Illustration/*.ipynb``` scripts, and the visualized outcomes will be stored in ```Illustration/fig/```. Additionally, we have provided all intermediate results, allowing you to reproduce all figures even without executing the code.

**Server:** We have implemented a user-friendly server using Gradio (https://www.gradio.app/), which you can access through the following link: https://huggingface.co/spaces/yjwtheonly/Scorpius_HF. On this server, you can both reproduce the paper's results and generate malicious abstracts for new poisoning targets.