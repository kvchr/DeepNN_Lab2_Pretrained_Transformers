"""Lab 2: fine-tuning a pretrained transformer (Yelp polarity) and using pretrained models on new tasks
(PIQA, TruthfulQA, WinoGrande) without further training.

The notebook `lab2_transformers_nlp.ipynb` calls these functions. With smoke=True every dataset and model is
replaced by a tiny random stand-in, so the code can be tested on a CPU without downloads.
"""
import hashlib
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Dataset files on the Hugging Face Hub (repository, file, pinned revision)
DATASETS = {
    "yelp_train": ("fancyzhx/yelp_polarity", "plain_text/train-00000-of-00001.parquet",
                   "bbf1c97a1f0cf005e5aded43839fd814654a1557"),
    "yelp_test": ("fancyzhx/yelp_polarity", "plain_text/test-00000-of-00001.parquet",
                  "bbf1c97a1f0cf005e5aded43839fd814654a1557"),
    "piqa": ("baber/piqa", "piqa_validation.parquet", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1"),
    "winogrande": ("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet",
                   "01e74176c63542e6b0bcb004dcdea22d94fb67b5"),
    "truthfulqa": ("truthfulqa/truthful_qa", "generation/validation-00000-of-00001.parquet",
                   "741b8276f2d1982aa3d5b832d3ee81ed3b896490"),
}

SENTIMENT_BASE = "distilbert/distilbert-base-uncased"
SENTIMENT_SST2 = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MC_HEADS = ["danlou/roberta-large-finetuned-csqa", "LIAMF-USP/roberta-large-finetuned-race"]
CAUSAL_LMS = ["openai-community/gpt2", "openai-community/gpt2-medium", "openai-community/gpt2-large",
              "EleutherAI/pythia-1b", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
SENTENCE_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2",
                   "sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-distilroberta-v1",
                   "BAAI/bge-small-en-v1.5", "thenlper/gte-base"]
WINOGRANDE_CLASSIFIER = "DeepPavlov/roberta-large-winogrande"


def short(name):
    return name.split("/")[-1]


# ----------------------------------------------------------------------------- data

def _smoke_frames(name):
    rng = np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest()[:8], 16))
    good, bad = ["great", "tasty", "friendly", "lovely", "fresh"], ["awful", "rude", "cold", "dirty", "slow"]
    filler = ["the", "food", "was", "and", "service", "place", "very", "we", "had", "a", "meal"]
    if name.startswith("yelp"):
        n = 400 if name == "yelp_train" else 120
        labels = rng.integers(0, 2, n)
        text = [" ".join(rng.choice(filler, 6).tolist() + rng.choice(good if y else bad, 2).tolist()) for y in labels]
        return pd.DataFrame({"text": text, "label": labels})
    if name == "piqa":
        return pd.DataFrame({"goal": [f"how do i {w} the meal" for w in rng.choice(filler, 24)],
                             "sol1": ["use a fresh plate"] * 24, "sol2": ["use a cold jar"] * 24,
                             "label": rng.integers(0, 2, 24)})
    if name == "winogrande":
        return pd.DataFrame({"sentence": ["the food was better than the meal so _ was eaten first ."] * 24,
                             "option1": ["food"] * 24, "option2": ["meal"] * 24,
                             "answer": rng.choice(["1", "2"], 24)})
    if name == "truthfulqa":
        rows = []
        for i in range(24):
            rows.append(dict(question=f"what was the food like {i}", best_answer="the food was fresh",
                             correct_answers=np.array(["the food was fresh", "it was fresh", "fresh and tasty",
                                                       "very fresh"][: 2 + i % 3]),
                             incorrect_answers=np.array(["the food was cold", "it was awful"])))
        return pd.DataFrame(rows)
    raise KeyError(name)


def load_frame(name, smoke=False):
    """A dataset file as a pandas DataFrame (downloaded once into the Hugging Face cache)."""
    if smoke:
        return _smoke_frames(name)
    from huggingface_hub import hf_hub_download
    repo, filename, revision = DATASETS[name]
    return pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def clean_review(t):
    return t.replace("\\n", " ").replace('\\"', '"').strip()


def yelp_splits(smoke=False, seed=1234, n_train=150, n_val=500, n_test=1000):
    """Class-balanced samples: train and validation from the training split (disjoint), test from the test split.
    Label 0 = negative review (1-2 stars), 1 = positive review (3-4 stars)."""
    if smoke:
        n_train, n_val, n_test = 12, 16, 16
    tr, te = load_frame("yelp_train", smoke), load_frame("yelp_test", smoke)
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(tr["label"].to_numpy() == c))
        parts["train"].append(tr.iloc[idx[:n_train]])
        parts["val"].append(tr.iloc[idx[n_train:n_train + n_val]])
        parts["test"].append(te.iloc[rng.permutation(np.flatnonzero(te["label"].to_numpy() == c))[:n_test]])
    out = {}
    for k, v in parts.items():
        df = pd.concat(v).sample(frac=1, random_state=seed).reset_index(drop=True)
        out[k] = pd.DataFrame({"text": df["text"].map(clean_review), "label": df["label"].astype(int)})
    return out


def truthfulqa_items(smoke=False):
    """Questions with a best answer, two further correct answers that differ from it, and one incorrect answer.
    Candidates are always [best, correct 1, correct 2, incorrect]."""
    norm = lambda s: " ".join(str(s).lower().strip().rstrip(".").split())
    items = []
    for _, r in load_frame("truthfulqa", smoke).iterrows():
        best = str(r["best_answer"]).strip()
        others, seen = [], {norm(best)}
        for c in r["correct_answers"]:
            if norm(c) not in seen:
                seen.add(norm(c))
                others.append(str(c).strip())
        if len(others) >= 2 and len(r["incorrect_answers"]) >= 1:
            items.append(dict(question=str(r["question"]).strip(), choices=[best, others[0], others[1],
                                                                            str(r["incorrect_answers"][0]).strip()]))
    return items


# ----------------------------------------------------------------------------- models

def _smoke_tokenizer(kind):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    words = ("the food was and service place very we had a meal great tasty friendly lovely fresh awful rude "
             "cold dirty slow how do i use plate jar better than so eaten first it what like question answer "
             "true false . , ? : _ yes no").split()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[EOS]"]
    vocab = {w: i for i, w in enumerate(specials + words + [str(i) for i in range(30)])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace()])
    if kind != "lm":
        tk.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                  sep_token="[SEP]", eos_token="[EOS]")
    tok.model_max_length = 512
    return tok


def _smoke_model(kind, name):
    import transformers as T
    torch.manual_seed(int(hashlib.md5(name.encode()).hexdigest()[:8], 16))
    tok = _smoke_tokenizer(kind)
    v, pad = len(tok), tok.pad_token_id
    if kind == "lm":
        model = T.GPT2LMHeadModel(T.GPT2Config(vocab_size=v, n_embd=32, n_layer=1, n_head=2, n_positions=256,
                                               pad_token_id=pad))
    elif kind == "cls":
        model = T.DistilBertForSequenceClassification(T.DistilBertConfig(vocab_size=v, dim=32, hidden_dim=64,
                                                                         n_layers=1, n_heads=2, pad_token_id=pad))
    else:
        cfg = T.BertConfig(vocab_size=v, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=64, pad_token_id=pad, num_labels=2,
                           id2label={0: "False", 1: "True"}, label2id={"False": 0, "True": 1})
        model = T.BertForMultipleChoice(cfg) if kind == "mc" else T.BertForSequenceClassification(cfg)
    return tok, model


def load_model(kind, name, device, smoke=False):
    """kind: 'cls' (sequence classification), 'pair' (sentence-pair classification), 'mc' (multiple choice), 'lm'."""
    if smoke:
        tok, model = _smoke_model(kind, name)
    else:
        import transformers as T
        tok = T.AutoTokenizer.from_pretrained(name)
        auto = {"cls": T.AutoModelForSequenceClassification, "pair": T.AutoModelForSequenceClassification,
                "mc": T.AutoModelForMultipleChoice, "lm": T.AutoModelForCausalLM}[kind]
        kw = {"num_labels": 2} if name == SENTIMENT_BASE else {}
        model = auto.from_pretrained(name, dtype=torch.float32, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, model.to(device).eval()


class _SmokeSentenceModel:
    def __init__(self, name):
        self.salt = name

    def encode(self, sentences, normalize_embeddings=True, convert_to_numpy=True, batch_size=64,
               show_progress_bar=False):
        out = np.zeros((len(sentences), 64))
        for i, s in enumerate(sentences):
            for w in s.lower().split():
                out[i, int(hashlib.md5((self.salt + w).encode()).hexdigest()[:6], 16) % 64] += 1
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)


def load_sentence_model(name, device, smoke=False):
    if smoke:
        return _SmokeSentenceModel(name)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=str(device))


def free_model(name, smoke=False):
    """Delete a model's files from the Hugging Face cache (keeps the notebook's disk use small)."""
    if smoke:
        return
    from huggingface_hub import constants
    shutil.rmtree(Path(constants.HF_HUB_CACHE) / ("models--" + name.replace("/", "--")), ignore_errors=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------- part A: fine-tuning

def _pad(seqs, pad_id, device):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def classify(model, ids, pad_id, device, batch=64):
    """Predicted class for each token-id sequence (sequences sorted by length so padding stays small)."""
    model.eval()
    order = np.argsort([len(s) for s in ids])
    preds = np.empty(len(ids), dtype=int)
    for s in range(0, len(ids), batch):
        sel = order[s:s + batch]
        x, m = _pad([ids[i] for i in sel], pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            preds[sel] = model(input_ids=x, attention_mask=m).logits.argmax(-1).cpu().numpy()
    return preds


def finetune(tok, enc, labels, lr, batch, epochs, seed, device, smoke=False, warmup=0.1, weight_decay=0.01):
    """Fine-tune a fresh copy of the pretrained model: AdamW, linear warm-up (10%) and decay, mixed precision.
    Returns validation accuracy after every epoch and the final test accuracy."""
    from transformers import get_linear_schedule_with_warmup, set_seed
    set_seed(seed)
    _, model = load_model("cls", SENTIMENT_BASE, device, smoke)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [{"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
               "weight_decay": weight_decay},
              {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr)
    n = len(enc["train"])
    steps = math.ceil(n / batch) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    cuda = str(device).startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    g = torch.Generator().manual_seed(seed)
    y_train = torch.tensor(labels["train"])
    hist = dict(train_loss=[], val_acc=[])
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).tolist()
        loss_sum = 0.0
        for s in range(0, n, batch):
            sel = perm[s:s + batch]
            x, m = _pad([enc["train"][i] for i in sel], tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss = F.cross_entropy(model(input_ids=x, attention_mask=m).logits.float(), y_train[sel].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(sel)
        hist["train_loss"].append(loss_sum / n)
        hist["val_acc"].append(float((classify(model, enc["val"], tok.pad_token_id, device) == labels["val"]).mean()))
    test_acc = float((classify(model, enc["test"], tok.pad_token_id, device) == labels["test"]).mean())
    del model
    if cuda:
        torch.cuda.empty_cache()
    return dict(lr=lr, batch=batch, seed=seed, epochs=epochs, updates=steps, history=hist,
                val_acc=hist["val_acc"][-1], test_acc=test_acc, seconds=time.time() - t0)


# ----------------------------------------------------------------------------- part B: scoring without training

@torch.no_grad()
def lm_loglik(model, tok, pairs, device, batch=16, max_len=1024):
    """Log-probability of each continuation given its context: sum over the continuation's tokens.
    Contexts and continuations are split as in lm-evaluation-harness (trailing spaces move to the continuation)."""
    reqs = []
    for ctx, cont in pairs:
        n_sp = len(ctx) - len(ctx.rstrip())
        if n_sp:
            ctx, cont = ctx[:-n_sp], ctx[-n_sp:] + cont
        c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        whole = tok(ctx + cont, add_special_tokens=False)["input_ids"]
        cont_ids = whole[len(c_ids):] if whole[:len(c_ids)] == c_ids else tok(cont, add_special_tokens=False)["input_ids"]
        ids = (c_ids + cont_ids)[-max_len:]
        reqs.append((ids, len(cont_ids)))
    out = np.zeros(len(reqs))
    order = np.argsort([-len(r[0]) for r in reqs])
    for s in range(0, len(reqs), batch):
        sel = order[s:s + batch]
        x, m = _pad([reqs[i][0] for i in sel], tok.pad_token_id, device)
        logp = F.log_softmax(model(input_ids=x, attention_mask=m).logits.float(), -1)
        for row, i in enumerate(sel):
            ids, k = reqs[i]
            L = len(ids)
            targets = torch.tensor(ids[L - k:], device=device)
            out[i] = logp[row, L - k - 1:L - 1].gather(1, targets[:, None]).sum().item()
    return out


@torch.no_grad()
def mc_logits(model, tok, firsts, choices, device, batch=8, max_len=256):
    """Multiple-choice head: one score per (first text, choice) pair; choices is a list of equal-length lists."""
    n_c = len(choices[0])
    out = np.zeros((len(firsts), n_c))
    for s in range(0, len(firsts), batch):
        fs, cs = firsts[s:s + batch], choices[s:s + batch]
        a = [f for f in fs for _ in range(n_c)]
        b = [c for ch in cs for c in ch]
        enc = tok(a, b, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.view(len(fs), n_c, -1).to(device) for k, v in enc.items()}
        out[s:s + len(fs)] = model(**enc).logits.float().cpu().numpy()
    return out


@torch.no_grad()
def pair_true_prob(model, tok, firsts, seconds, device, batch=32, max_len=256):
    """Sentence-pair classifier with labels False/True: probability of True for each pair."""
    true_id = {v.lower(): k for k, v in model.config.id2label.items()}["true"]
    out = np.zeros(len(firsts))
    for s in range(0, len(firsts), batch):
        enc = tok(firsts[s:s + batch], seconds[s:s + batch], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        out[s:s + batch] = F.softmax(model(**enc).logits.float(), -1)[:, int(true_id)].cpu().numpy()
    return out


def piqa_scores(kind, model, tok, df, device):
    """Score of each of the two solutions; for language models also the score per character of the solution."""
    if kind == "mc":
        sc = mc_logits(model, tok, df["goal"].tolist(), df[["sol1", "sol2"]].values.tolist(), device)
        return sc, sc
    pairs = [(f"Question: {g}\nAnswer:", f" {s}")
             for g, s1, s2 in df[["goal", "sol1", "sol2"]].values for s in (s1, s2)]
    ll = lm_loglik(model, tok, pairs, device).reshape(-1, 2)
    lengths = np.array([[len(s1), len(s2)] for s1, s2 in df[["sol1", "sol2"]].values])
    return ll, ll / lengths


def truthfulqa_scores(kind, model, tok, items, device):
    if kind == "mc":
        return mc_logits(model, tok, [it["question"] for it in items], [it["choices"] for it in items], device)
    pairs = [(f"Q: {it['question']}\nA:", f" {c}") for it in items for c in it["choices"]]
    return lm_loglik(model, tok, pairs, device).reshape(-1, 4)


def winogrande_split(sentence):
    i = sentence.index("_")
    return sentence[:i], sentence[i + 1:]


def winogrande_scores(kind, model, tok, df, device):
    """Two scores per sentence, one for each option filling the blank '_'."""
    pre, post = zip(*[winogrande_split(s) for s in df["sentence"]])
    opts = df[["option1", "option2"]].values.tolist()
    if kind == "pair":   # as the classifier was trained: (text before the blank, option + text after the blank)
        firsts = [p for p in pre for _ in range(2)]
        seconds = [o + q for o_pair, q in zip(opts, post) for o in o_pair]
        return pair_true_prob(model, tok, firsts, seconds, device).reshape(-1, 2)
    if kind == "mc":
        return mc_logits(model, tok, list(pre), [[o + q for o in o_pair] for o_pair, q in zip(opts, post)], device)
    # partial scoring: probability of the text after the blank, given the text before it with the option filled in
    pairs = [(p + o, " " + q.strip()) for p, o_pair, q in zip(pre, opts, post) for o in o_pair]
    return lm_loglik(model, tok, pairs, device).reshape(-1, 2)


def accuracy_ci(correct):
    """Accuracy with a 95% Wilson interval."""
    c = np.asarray(correct, dtype=float)
    n, p, z = len(c), c.mean(), 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(p), float(centre - half), float(centre + half)
