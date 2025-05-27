import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

# === Model Paths ===
paths = []

# === Indices to REMOVE from the first model's classifier
extra_indices = [1, 3, 14, 16, 17]  # these will be excluded
all_indices = list(range(18))  # v2_2 has 18 outputs
keep_indices = [i for i in all_indices if i not in extra_indices]

# === Load State Dicts ===
state_dicts = []
for i, path in enumerate(paths):
    model = RobertaForSequenceClassification.from_pretrained(path)
    state_dict = model.state_dict()
    
    if state_dict['classifier.out_proj.weight'].shape[0] == 18:
        print(f"Initial shape of classifier.out_proj.weight (model {i}): {state_dict['classifier.out_proj.weight'].shape}")
        print(f"Initial shape of classifier.out_proj.bias (model {i}): {state_dict['classifier.out_proj.bias'].shape}")
        
        # Trim classifier to only valid indices
        state_dict["classifier.out_proj.weight"] = state_dict["classifier.out_proj.weight"][keep_indices]
        state_dict["classifier.out_proj.bias"] = state_dict["classifier.out_proj.bias"][keep_indices]
        
        # Check shapes after trimming
        print(f"Trimmed shape of classifier.out_proj.weight (model {i}): {state_dict['classifier.out_proj.weight'].shape}")
        print(f"Trimmed shape of classifier.out_proj.bias (model {i}): {state_dict['classifier.out_proj.bias'].shape}")
        
    state_dicts.append(state_dict)

# === Average Encoder and Classifier (only 13 classes) ===
def average_tensors(tensors):
    # Make sure tensors are of the same shape before averaging
    assert all(tensor.shape == tensors[0].shape for tensor in tensors), "Tensors have mismatched shapes"
    return sum(tensors) / len(tensors)

def average_tensors(tensors):
    return sum(tensors) / len(tensors)

merged_state = {}
for key in state_dicts[0].keys():
    if "classifier.out_proj.weight" in key or "classifier.out_proj.bias" in key:
        # Classifier weights: correctly average across models
        weights = [sd[key] for sd in state_dicts]
        merged_state[key] = torch.stack(weights).mean(dim=0)
    else:
        # Normal weights: average across all 3 models
        tensors = [sd[key] for sd in state_dicts]
        merged_state[key] = average_tensors(tensors)

# === Create new model with merged weights (13 labels) ===
student_model = RobertaForSequenceClassification.from_pretrained(
    "youscan/ukr-roberta-base", num_labels=13
)
student_model.load_state_dict(merged_state)

# === Save final merged model ===
types = [
    "Fearmongering",
    "Doubt Casting",
    "Slogan",
    "Flag Waving",
    "Loaded Language",
    "Demonizing the Enemy",
    "Name Calling",
    "Common Man",
    "Scapegoating",
    "Smear",
    "Virtue Words",
    "Conspiracy Theory",
    "Oversimplification"
]

save_path = "distilled_student_model_merged_complete_v1.3"
student_model.save_pretrained(save_path)

# Load tokenizer and save it (from any model, assuming they are the same)
tokenizer = RobertaTokenizer.from_pretrained(paths[0])
tokenizer.save_pretrained(save_path)

# Adjust config manually and save it
config = RobertaConfig.from_pretrained(paths[0])
config.num_labels = 13
config.id2label = {i: label for i, label in enumerate(types)}
config.label2id = {label: i for i, label in enumerate(types)}
config.save_pretrained(save_path)
