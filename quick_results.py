from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import numpy as np

print("=== QUICK STS-B + Mini-QQP ===")

sts = load_dataset("glue", "stsb")["validation"]
sts_s1, sts_s2, sts_labels = sts["sentence1"], sts["sentence2"], np.array(sts["label"], dtype=float)

qqp_mini = load_dataset("glue", "qqp")["train"].select(range(1000))
q1_mini, q2_mini, y_mini = qqp_mini["question1"], qqp_mini["question2"], qqp_mini["label"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load models
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def bert_mean(texts):
    bert_model.eval()
    with torch.no_grad():
        enc = bert_tokenizer(texts[:512], padding=True, truncation=True, return_tensors="pt").to(device)  # Limit size
        outputs = bert_model(**enc)
        mask = enc.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        summed = torch.sum(outputs.last_hidden_state * mask, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        return (summed / summed_mask).cpu()

print("Encoding STS-B...")
bert_mean_s1_sts = bert_mean(sts_s1)
bert_mean_s2_sts = bert_mean(sts_s2)
sbert_s1_sts = sbert_model.encode(sts_s1[:512])
sbert_s2_sts = sbert_model.encode(sts_s2[:512])

sims_bert_sts = cosine_similarity(bert_mean_s1_sts, bert_mean_s2_sts).diagonal()
sims_sbert_sts = cosine_similarity(sbert_s1_sts, sbert_s2_sts).diagonal()
sims_scaled_sts = (sims_sbert_sts + 1) * 2.5

p_bert_sts, s_bert_sts = pearsonr((cosine_similarity(bert_mean_s1_sts, bert_mean_s2_sts).diagonal() + 1) * 2.5, sts_labels[:512]), spearmanr((cosine_similarity(bert_mean_s1_sts, bert_mean_s2_sts).diagonal() + 1) * 2.5, sts_labels[:512])
p_sbert_sts, s_sbert_sts = pearsonr((sims_sbert_sts + 1) * 2.5, sts_labels[:512]), spearmanr((sims_sbert_sts + 1) * 2.5, sts_labels[:512
