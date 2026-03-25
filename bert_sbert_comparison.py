from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

print("=== SBERT vs BERT Empirical Study ===")
print("Loading datasets...")

# STS-B (1,500 validation pairs)
sts = load_dataset("glue", "stsb")["validation"]
sts_s1, sts_s2, sts_labels = sts["sentence1"], sts["sentence2"], np.array(sts["label"], dtype=float)

# QQP subset (20k total)
print("Preparing QQP subset...")
qqp = load_dataset("glue", "qqp")
qqp_data = [(q1,q2,l) for q1,q2,l in zip(qqp["train"]["question1"][:20000], 
                                         qqp["train"]["question2"][:20000], 
                                         qqp["train"]["label"][:20000]) if q1 and q2]
np.random.seed(42)
np.random.shuffle(qqp_data)
train_size, valid_size, test_size = 12000, 4000, 4000
q1_train, q2_train, y_train = zip(*qqp_data[:train_size])
q1_valid, q2_valid, y_valid = zip(*qqp_data[train_size:train_size+valid_size])
q1_test, q2_test, y_test = zip(*qqp_data[train_size+valid_size:])

print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

bert_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_name)
bert_model = AutoModel.from_pretrained(bert_name).to(device)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def bert_cls(texts, batch_size=32):
    bert_model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = bert_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = bert_model(**enc)
            embeddings.append(outputs.last_hidden_state[:,0].cpu())
    return torch.cat(embeddings, dim=0)

def bert_mean(texts, batch_size=32):
    bert_model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = bert_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            outputs = bert_model(**enc)
            mask = enc.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            embeddings.append((summed / summed_mask).cpu())
    return torch.cat(embeddings, dim=0)

print("\n--- STS-B Experiment ---")
print("Encoding STS-B...")

# Encode STS-B
bert_cls_s1 = bert_cls(sts_s1)
bert_cls_s2 = bert_cls(sts_s2)
bert_mean_s1 = bert_mean(sts_s1)
bert_mean_s2 = bert_mean(sts_s2)
sbert_s1 = sbert_model.encode(sts_s1, batch_size=32, show_progress_bar=False)
sbert_s2 = sbert_model.encode(sts_s2, batch_size=32, show_progress_bar=False)

def correlations(emb1, emb2, labels):
    sims = cosine_similarity(emb1, emb2).diagonal()
    sims_scaled = (sims + 1) * 2.5  # -1..1 -> 0..5
    pearson = pearsonr(sims_scaled, labels)[0]
    spearman = spearmanr(sims_scaled, labels)[0]
    return pearson, spearman

print("\nSTS-B Results:")
results = {}
for name, e1, e2 in [("BERT-CLS", bert_cls_s1, bert_cls_s2), 
                     ("BERT-mean", bert_mean_s1, bert_mean_s2),
                     ("SBERT", sbert_s1, sbert_s2)]:
    p, s = correlations(e1, e2, sts_labels)
    results[name] = {"pearson": p, "spearman": s}
    print(f"{name:10}: Pearson={p:.3f}, Spearman={s:.3f}")

print("\n--- QQP Experiment ---")
print("Encoding QQP...")

# Encode QQP
for split in ["train", "valid", "test"]:
    q1s, q2s = locals()[f"q1_{split}"], locals()[f"q2_{split}"]
    print(f"  Encoding {split}...")
    globals()[f"bert_mean_q1_{split}"] = bert_mean(q1s)
    globals()[f"bert_mean_q2_{split}"] = bert_mean(q2s)
    globals()[f"sbert_q1_{split}"] = sbert_model.encode(q1s, batch_size=32, show_progress_bar=False)
    globals()[f"sbert_q2_{split}"] = sbert_model.encode(q2s, batch_size=32, show_progress_bar=False)

def best_threshold(scores, labels):
    best_f1, best_t = 0, 0
    for t in np.arange(-1, 1, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1: 
            best_f1, best_t = f1, t
    return best_t, best_f1

print("\nQQP Results:")
qqp_results = {}
for model_name, model_prefix in [("BERT-mean", "bert_mean"), ("SBERT", "sbert")]:
    scores_valid = cosine_similarity(globals()[f"{model_prefix}_q1_valid"], 
                                   globals()[f"{model_prefix}_q2_valid"]).diagonal()
    t, f1_valid = best_threshold(scores_valid, y_valid)
    scores_test = cosine_similarity(globals()[f"{model_prefix}_q1_test"], 
                                  globals()[f"{model_prefix}_q2_test"]).diagonal()
    preds_test = (scores_test >= t).astype(int)
    acc_test = accuracy_score(y_test, preds_test)
    f1_test = f1_score(y_test, preds_test)
    
    qqp_results[model_name] = {"threshold": t, "valid_f1": f1_valid, "test_acc": acc_test, "test_f1": f1_test}
    print(f"{model_name:10}: threshold={t:.3f}, valid F1={f1_valid:.3f}, test F1={f1_test:.3f}, test Acc={acc_test:.3f}")

# Save results for paper
with open("results.txt", "w") as f:
    f.write("STS-B Results:\n")
    f.write("Model\tPearson\tSpearman\n")
    for name, res in results.items():
        f.write(f"{name}\t{res['pearson']:.3f}\t{res['spearman']:.3f}\n")
    
    f.write("\nQQP Results:\n")
    f.write("Model\tThreshold\tValid F1\tTest Acc\tTest F1\n")
    for name, res in qqp_results.items():
        f.write(f"{name}\t{res['threshold']:.3f}\t{res['valid_f1']:.3f}\t{res['test_acc']:.3f}\t{res['test_f1']:.3f}\n")

print("\n✅ Results saved to results.txt")

