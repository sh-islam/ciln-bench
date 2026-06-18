"""IDN-degree comparison across (voter pool x subset filter) x voter-count.

Cells:
  A. Gu-arch pool (7 voters), clean-correct subset
  B. Gu-arch pool (7 voters), all NLT rows
  C. v2 pool       (4 voters), clean-correct subset
  D. v2 pool       (4 voters), all NLT rows
  E. Gu released noisy_labels low (10 voters), full release
  F. Gu released noisy_labels medium (10 voters), full release
  G. Gu released noisy_labels high (10 voters), full release

Voter-count modes per cell: native, sub4, sub7 (where applicable).

Metrics, per setting:
  m1: rate_variance  + CCN floor + gap
  m2: T_frobenius    + CCN floor + gap
  m3: partition_CMI on CORRUPTED CLIP partition + CCN floor + gap  (headline)
  m4: partition_CMI on CLEAN CLIP partition     + CCN floor + gap  (susceptibility)
  m5: kNN_CMI on corrupted CLIP embeddings      + bias floor + gap
  m6: kNN_CMI on clean CLIP embeddings          + bias floor + gap
  Plus: noise_rate (mean voter disagreement)

All metrics computed on per-image bincount distributions p(x) = bincount(voter_argmaxes)/N.

For Gu cells (E,F,G): X = clean CIFAR-10 train images (Gu corrupts labels only).
For our cells (A,B,C,D): X = clean image (for m4, m6), X_tilde = corrupted image (for m3, m5).

CCN floor: simulate per-image bincount under CCN at matched per-class rates and
voter count, recompute metric. Mean of 5 sims.

Partition: K=5 KMeans clusters per true class (50 total) on CLIP ViT-B/32.

Output: idn_2x2/results/<cell>__<voter_count>.json
"""
from __future__ import annotations
import gc, glob, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torchvision
import tfrecord
import clip
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

JOURNAL_ROOT = Path('/home/misla2/thesis/python/mnist_corruptions/journal_edition')
GU_NEW_POOL = JOURNAL_ROOT / 'gu_arch_voters' / 'inference' / 'per_setting'
GU_CC_DIR = JOURNAL_ROOT / 'gu_arch_voters' / 'inference' / 'per_setting_clean'
CIFAR_SOURCE = JOURNAL_ROOT / 'output_seed0' / 'cifar10'
CLEAN_NLT = JOURNAL_ROOT / 'output_seed0' / 'clean' / 'cifar10' / 'noisylabeltrain_clean'
GU_REL = JOURNAL_ROOT / 'gu_compare' / 'data' / 'cifar10'
SPLITS_DIR = JOURNAL_ROOT / 'output_seed0' / 'splits' / 'cifar10'
OUT = JOURNAL_ROOT / 'idn_2x2' / 'results'
OUT.mkdir(exist_ok=True, parents=True)

V2_VOTERS = ["resnet20", "wrn28_10", "deit3_small", "clip"]
GU_ARCH_VOTERS = ["mobilenetv1", "mobilenetv2", "vgg16", "resnet50", "resnet101", "nasnetmobile", "inception_v4"]
N_CLASSES = 10
N_CLUSTERS_PER_CLASS = 5
CCN_SIMS = 5
KNN_K = 3
RNG = np.random.default_rng(0)


# ============================================================
# Bincount distribution from voter labels
# ============================================================
def bincount_dist(votes, n_classes=N_CLASSES):
    """votes: (n_images, n_voters) of int class indices.
    Returns: (n_images, n_classes) float, each row sums to 1."""
    n_images, n_voters = votes.shape
    p = np.zeros((n_images, n_classes), dtype=np.float64)
    for k in range(n_voters):
        np.add.at(p, (np.arange(n_images), votes[:, k]), 1.0)
    p /= n_voters
    return p


# ============================================================
# Metrics
# ============================================================
def m1_rate_variance(p, true_y):
    """eta(x) = 1 - p(x)[true_y(x)]. Var within each true class y, mean over y."""
    eta = 1.0 - p[np.arange(len(p)), true_y]
    vals = []
    for y in range(N_CLASSES):
        m = true_y == y
        if m.sum() < 2: continue
        vals.append(float(np.var(eta[m])))
    return float(np.mean(vals))


def m2_frobenius(p, true_y):
    """E_x[ ||p(x) - p_bar_{y(x)}||^2 ]."""
    p_bar = np.zeros((N_CLASSES, N_CLASSES))
    for y in range(N_CLASSES):
        m = true_y == y
        if m.any():
            p_bar[y] = p[m].mean(axis=0)
    diff = p - p_bar[true_y]
    sq = (diff * diff).sum(axis=1)
    return float(sq.mean())


def m3_partition_cmi(p, true_y, partition):
    """KL(p_bar_g || p_bar_y) per group, averaged within class then across classes.
    partition: (n_images,) int. Each image's group id."""
    p_bar_y = np.zeros((N_CLASSES, N_CLASSES))
    for y in range(N_CLASSES):
        m = true_y == y
        if m.any():
            p_bar_y[y] = p[m].mean(axis=0)
    eps = 1e-9
    out_per_class = []
    for y in range(N_CLASSES):
        m = true_y == y
        if not m.any(): continue
        ref = p_bar_y[y] + eps
        ref /= ref.sum()
        groups_in_class = np.unique(partition[m])
        kls = []
        for g in groups_in_class:
            mg = (true_y == y) & (partition == g)
            if not mg.any(): continue
            p_bar_g = p[mg].mean(axis=0) + eps
            p_bar_g /= p_bar_g.sum()
            kl = float((p_bar_g * np.log(p_bar_g / ref)).sum())
            kls.append(kl)
        if kls:
            out_per_class.append(float(np.mean(kls)))
    return float(np.mean(out_per_class)) if out_per_class else 0.0


def m5_knn_cmi(emb, votes_argmax, true_y, k=KNN_K, max_n=8000):
    """Ross 2014 kNN estimator of I(X; \tilde Y | Y).
    emb: (N, d) continuous. votes_argmax: (N,) discrete majority/argmax voter label.
    true_y: (N,) discrete true class.

    Estimator (Frenzel-Pompe style for CMI):
      I(X; T | Y) = digamma(k) + <digamma(N_y)> - <digamma(N_{x,y})> - <digamma(N_{t,y})>
    where N_y, N_{x,y}, N_{t,y} are counts within the k-NN epsilon-ball in (emb space)
    restricted to matching Y, matching (X,Y), matching (T,Y) respectively.

    Simpler approximation we use (works well enough for finite samples):
      Per-class I(X; T) using the kNN-MI estimator (Ross 2014), averaged with class
      weighting p(Y=y). For each class y, run kNN MI between emb[y-class subset] and
      votes_argmax[y-class subset]. Average weighted by class fraction.

    If n is too large we subsample to max_n for speed.
    """
    if len(emb) > max_n:
        idx = RNG.choice(len(emb), size=max_n, replace=False)
        emb = emb[idx]; votes_argmax = votes_argmax[idx]; true_y = true_y[idx]
    out_per_class = []
    for y in range(N_CLASSES):
        m = true_y == y
        n = int(m.sum())
        if n < k + 5: continue
        X_y = emb[m]
        T_y = votes_argmax[m]
        if len(np.unique(T_y)) < 2:
            out_per_class.append(0.0); continue
        mi = _ross_mi_continuous_discrete(X_y, T_y, k=k)
        out_per_class.append(max(0.0, float(mi)))
    return float(np.mean(out_per_class)) if out_per_class else 0.0


def _ross_mi_continuous_discrete(X, T, k=3):
    """Ross 2014 estimator: I(X; T) where X continuous, T discrete.
    I = digamma(N) - <digamma(N_t)> + digamma(k) - <digamma(m_i)>
    where m_i is the count of points in the same class as point i that are within
    distance epsilon_i, and epsilon_i is the distance to the k-th nearest neighbor
    of point i within its class.
    """
    N = len(X)
    if N == 0: return 0.0
    classes, counts = np.unique(T, return_counts=True)
    if len(classes) < 2: return 0.0
    sum_dig_nt = float(np.sum([counts[i] * digamma(counts[i]) for i in range(len(classes))])) / N
    # For each point, find its k-th nearest neighbor among same-class points
    # then count overall (any-class) points within that distance.
    log_term = 0.0
    nbrs_all = NearestNeighbors(n_neighbors=min(N, 200), n_jobs=-1).fit(X)
    for cls_idx, c in enumerate(classes):
        idx_c = np.where(T == c)[0]
        if len(idx_c) <= k: continue
        nbrs_c = NearestNeighbors(n_neighbors=k+1, n_jobs=-1).fit(X[idx_c])
        d_c, _ = nbrs_c.kneighbors(X[idx_c])
        eps = d_c[:, k]  # k-th NN distance within class (excluding self)
        # m_i = number of points in all data within distance eps (excluding self)
        d_all, _ = nbrs_all.kneighbors(X[idx_c], n_neighbors=min(N, 200))
        m_i = np.array([np.sum(d_all[i, 1:] < eps[i]) for i in range(len(idx_c))], dtype=float)
        m_i = np.maximum(m_i, 1.0)
        log_term += float(np.sum(digamma(m_i)))
    log_term /= N
    return digamma(N) - sum_dig_nt + digamma(k) - log_term


# ============================================================
# CCN floor (m1, m2, m3 only): simulate CCN at matched per-class rates
# ============================================================
def ccn_floors(p, true_y, partition, n_voters, sims=CCN_SIMS):
    """Simulate per-image bincount under CCN at per-class rate.
    Per class y: T_bar_y = mean p(x) for x in class y (a 10-vector, our 'class transition').
    Simulate n_voters draws per image from T_bar_{y(x)}, build bincount distribution,
    recompute m1, m2, m3. Return mean of (m1, m2, m3) across sims.
    """
    p_bar_y = np.zeros((N_CLASSES, N_CLASSES))
    for y in range(N_CLASSES):
        m = true_y == y
        if m.any():
            p_bar_y[y] = p[m].mean(axis=0)
    floors = {'m1': [], 'm2': [], 'm3': []}
    n_images = len(p)
    for s in range(sims):
        rng = np.random.default_rng(100 + s)
        sim_p = np.zeros((n_images, N_CLASSES), dtype=np.float64)
        for y in range(N_CLASSES):
            idx_y = np.where(true_y == y)[0]
            if len(idx_y) == 0: continue
            T_y = p_bar_y[y]
            T_y = T_y / T_y.sum() if T_y.sum() > 0 else np.ones(N_CLASSES) / N_CLASSES
            draws = rng.choice(N_CLASSES, size=(len(idx_y), n_voters), p=T_y)
            sim_p_y = bincount_dist(draws)
            sim_p[idx_y] = sim_p_y
        floors['m1'].append(m1_rate_variance(sim_p, true_y))
        floors['m2'].append(m2_frobenius(sim_p, true_y))
        floors['m3'].append(m3_partition_cmi(sim_p, true_y, partition))
    return {k: float(np.mean(v)) for k, v in floors.items()}


# ============================================================
# kNN CMI bias floor: simulate CCN, run estimator on it
# ============================================================
def knn_floor(emb_clean, emb_corr, true_y, p, n_voters, sims=2):
    """Simulate CCN, take argmax of bincount as voter label, run estimator."""
    p_bar_y = np.zeros((N_CLASSES, N_CLASSES))
    for y in range(N_CLASSES):
        m = true_y == y
        if m.any():
            p_bar_y[y] = p[m].mean(axis=0)
    floors_clean = []; floors_corr = []
    for s in range(sims):
        rng = np.random.default_rng(200 + s)
        sim_votes_argmax = np.zeros(len(p), dtype=np.int64)
        for y in range(N_CLASSES):
            idx_y = np.where(true_y == y)[0]
            if len(idx_y) == 0: continue
            T_y = p_bar_y[y]
            T_y = T_y / T_y.sum() if T_y.sum() > 0 else np.ones(N_CLASSES) / N_CLASSES
            draws = rng.choice(N_CLASSES, size=(len(idx_y), n_voters), p=T_y)
            # Majority/argmax of bincount
            sim_p_y = bincount_dist(draws)
            sim_votes_argmax[idx_y] = sim_p_y.argmax(axis=1)
        floors_clean.append(m5_knn_cmi(emb_clean, sim_votes_argmax, true_y))
        if emb_corr is not None:
            floors_corr.append(m5_knn_cmi(emb_corr, sim_votes_argmax, true_y))
    return float(np.mean(floors_clean)), float(np.mean(floors_corr)) if floors_corr else 0.0


# ============================================================
# Partition builder (K-means per true class)
# ============================================================
def build_partition(emb, true_y, k=N_CLUSTERS_PER_CLASS):
    """Returns (n_images,) int partition id (global, k * N_CLASSES groups)."""
    n = len(emb)
    part = np.zeros(n, dtype=np.int64)
    for y in range(N_CLASSES):
        idx = np.where(true_y == y)[0]
        if len(idx) < k:
            part[idx] = y * k  # all into one group
            continue
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(emb[idx])
        part[idx] = y * k + km.labels_
    return part


# ============================================================
# CLIP encoder
# ============================================================
print("Loading CLIP...")
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

def encode_images(images_uint8, batch_size=512):
    """images_uint8: (N,32,32,3). Returns (N,512) float64."""
    out = np.zeros((len(images_uint8), 512), dtype=np.float64)
    for s in range(0, len(images_uint8), batch_size):
        e = min(s + batch_size, len(images_uint8))
        batch = torch.stack([clip_preprocess(Image.fromarray(images_uint8[i])) for i in range(s, e)]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(batch).cpu().numpy().astype(np.float64)
        out[s:e] = emb
    # L2-normalize
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out


# ============================================================
# Load CIFAR-10 train + indices
# ============================================================
print("Loading CIFAR-10 train + NLT indices...")
DATA_ROOT = JOURNAL_ROOT.parent / 'data'
train_ds = torchvision.datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=False)
train_imgs_all = train_ds.data
train_labels_all = np.array(train_ds.targets)
nlt_idx = np.load(SPLITS_DIR / 'noisylabeltrain_indices.npy')
nlt_images_clean = train_imgs_all[nlt_idx]
nlt_true_y = train_labels_all[nlt_idx]
print(f"NLT size: {len(nlt_idx)} images")

print("Encoding NLT clean images with CLIP (one-time)...")
t0 = time.time()
nlt_emb_clean = encode_images(nlt_images_clean)
print(f"  done in {time.time()-t0:.0f}s. shape={nlt_emb_clean.shape}")

# clean partition (used for m4 across all cells/settings; setting-independent)
clean_partition = build_partition(nlt_emb_clean, nlt_true_y)
print(f"  clean partition built ({N_CLUSTERS_PER_CLASS} per class x {N_CLASSES} classes = {N_CLUSTERS_PER_CLASS*N_CLASSES} groups)")


# ============================================================
# Load v2-pool clean-correct mask (build from clean softmaxes)
# ============================================================
print("Building v2-pool clean-correct mask...")
v2_clean_correct_mask = np.ones(len(nlt_idx), dtype=bool)
clean_lbl = np.load(CLEAN_NLT / 'labels.npy')
assert np.array_equal(clean_lbl, nlt_true_y), "v2 clean labels mismatch NLT true"
for v in V2_VOTERS:
    sm = np.load(CLEAN_NLT / f'softmax_{v}.npy')
    pred = sm.argmax(axis=1)
    v2_clean_correct_mask &= (pred == nlt_true_y)
print(f"  v2 clean-correct: {v2_clean_correct_mask.sum()}/{len(v2_clean_correct_mask)} = {v2_clean_correct_mask.mean()*100:.1f}%")

# Gu-arch clean-correct mask (pre-computed)
gu_clean_correct_mask = np.load(GU_CC_DIR / 'clean_correct_mask.npy')
print(f"  gu_arch clean-correct: {gu_clean_correct_mask.sum()}/{len(gu_clean_correct_mask)} = {gu_clean_correct_mask.mean()*100:.1f}%")


# ============================================================
# Per-setting list (all 45 CIFAR-10)
# ============================================================
SETTINGS = []
for c_dir in sorted(CIFAR_SOURCE.iterdir()):
    if not c_dir.is_dir(): continue
    for s_dir in sorted(c_dir.iterdir()):
        if not s_dir.is_dir(): continue
        sev = s_dir.name.split('_')[1]
        SETTINGS.append((f"{c_dir.name}_sev{sev}", c_dir, s_dir))
print(f"\n{len(SETTINGS)} CIFAR-10 settings")


# ============================================================
# Pool loaders: per-image (N voters) argmax
# ============================================================
def votes_v2(setting_dir):
    """v2 pool: 4 voters. Loads softmax_<voter>.npy, takes argmax."""
    nl_dir = setting_dir / 'noisy_label_train'
    out = []
    for v in V2_VOTERS:
        f = nl_dir / f'softmax_{v}.npy'
        if not f.exists(): return None
        out.append(np.load(f).argmax(axis=1))
    return np.stack(out, axis=1)  # (N, 4)


def votes_gu_arch(setting_name):
    """Gu-arch pool: 7 voters."""
    d = GU_NEW_POOL / setting_name
    out = []
    for v in GU_ARCH_VOTERS:
        f = d / f'softmax_{v}.npy'
        if not f.exists(): return None
        out.append(np.load(f).argmax(axis=1))
    return np.stack(out, axis=1)  # (N, 7)


# ============================================================
# Gu released noisy_labels loader (10 voters per image)
# ============================================================
GU_SCHEMA = {'image/raw':'byte','image/class/label':'int','noisy_labels':'int','rater_ids':'byte'}
def load_gu_released_votes(level):
    """Returns (votes (N,10), true_y (N,), image_hashes [N])."""
    out_v, out_t, out_h = [], [], []
    for shard in sorted(glob.glob(str(GU_REL / level / 'train-*'))):
        for ex in tfrecord.tfrecord_loader(shard, None, description=GU_SCHEMA):
            out_v.append(list(map(int, ex['noisy_labels'])))
            out_t.append(int(ex['image/class/label'][0]) if hasattr(ex['image/class/label'], '__len__') else int(ex['image/class/label']))
            out_h.append(hashlib.sha256(bytes(ex['image/raw'])).hexdigest())
    return np.array(out_v, dtype=np.int64), np.array(out_t, dtype=np.int64), out_h


# ============================================================
# Voter subsampling
# ============================================================
def subsample_voters(votes, n_target, seed=0):
    """votes: (N_img, N_voters). Pick first n_target voters in a fixed shuffle."""
    n_img, n_voters = votes.shape
    if n_voters <= n_target: return votes
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_voters)[:n_target]
    return votes[:, perm]


# ============================================================
# Per-setting runner
# ============================================================
def run_setting(setting_name, votes, true_y, emb_clean, emb_corr_or_none, partition_clean,
                voter_count_label, n_voters_for_floor):
    """Compute all metrics for one (cell x setting x voter_count)."""
    p = bincount_dist(votes)
    noise_rate = float((1.0 - p[np.arange(len(p)), true_y]).mean())

    # Partition on corrupted (for m3 headline) — fall back to clean if no corrupted emb
    if emb_corr_or_none is not None:
        partition_corr = build_partition(emb_corr_or_none, true_y)
    else:
        partition_corr = partition_clean  # Gu: no corrupted image

    m1 = m1_rate_variance(p, true_y)
    m2 = m2_frobenius(p, true_y)
    m3 = m3_partition_cmi(p, true_y, partition_corr)
    m4 = m3_partition_cmi(p, true_y, partition_clean)

    # CCN floors for m1, m2, m3 (use corrupted partition for m3 floor)
    floors_corr = ccn_floors(p, true_y, partition_corr, n_voters=n_voters_for_floor)
    floors_clean = ccn_floors(p, true_y, partition_clean, n_voters=n_voters_for_floor)
    # m1, m2 are partition-independent, use either; use corr_floors values for m1/m2
    m1_floor = floors_corr['m1']; m2_floor = floors_corr['m2']
    m3_floor = floors_corr['m3']; m4_floor = floors_clean['m3']

    # m5, m6: kNN CMI using voter-argmax (mode of voter labels per image)
    votes_argmax = p.argmax(axis=1)
    m5 = m5_knn_cmi(emb_corr_or_none if emb_corr_or_none is not None else emb_clean,
                    votes_argmax, true_y)
    m6 = m5_knn_cmi(emb_clean, votes_argmax, true_y)
    m5_bias, m6_bias = knn_floor(emb_clean,
                                  emb_corr_or_none if emb_corr_or_none is not None else None,
                                  true_y, p, n_voters=n_voters_for_floor)
    # If emb_corr is None we set m5 same as m6 effectively
    if emb_corr_or_none is None:
        m5_bias = m6_bias

    return {
        "name": setting_name,
        "voter_count": voter_count_label,
        "n_images": int(len(p)),
        "noise_rate": noise_rate,
        "m1_rate_variance": m1,
        "m1_ccn_floor": m1_floor,
        "m1_gap": m1 - m1_floor,
        "m2_frobenius": m2,
        "m2_ccn_floor": m2_floor,
        "m2_gap": m2 - m2_floor,
        "m3_partition_cmi_corr": m3,
        "m3_ccn_floor": m3_floor,
        "m3_gap": m3 - m3_floor,
        "m4_partition_cmi_clean": m4,
        "m4_ccn_floor": m4_floor,
        "m4_gap": m4 - m4_floor,
        "m5_knn_cmi_corr": m5,
        "m5_bias_floor": m5_bias,
        "m5_gap": m5 - m5_bias,
        "m6_knn_cmi_clean": m6,
        "m6_bias_floor": m6_bias,
        "m6_gap": m6 - m6_bias,
    }


# ============================================================
# Cells A-D: our pools (need to encode corrupted images per setting)
# ============================================================
def run_our_cells(pool_name, voters_loader_fn, n_native, cc_mask):
    """Run cells {clean-correct, all-NLT} x {native, sub4, sub7} for one pool."""
    results = {label: [] for label in [
        f"{pool_name}_cc_native", f"{pool_name}_cc_sub4",
        f"{pool_name}_all_native", f"{pool_name}_all_sub4",
    ]}
    if n_native >= 7:
        results[f"{pool_name}_cc_sub7"] = []
        results[f"{pool_name}_all_sub7"] = []

    for i, (name, c_dir, s_dir) in enumerate(SETTINGS):
        votes = voters_loader_fn(name, c_dir, s_dir)
        if votes is None:
            print(f"  [{i+1}/{len(SETTINGS)}] {name}: SKIP (no votes)")
            continue
        # Load + encode corrupted images
        img_path = s_dir / 'noisy_label_train' / 'images.npy'
        corr_images = np.load(img_path)
        t0 = time.time()
        emb_corr = encode_images(corr_images)
        enc_t = time.time() - t0

        # ---- ALL-NLT (no clean-correct filter) ----
        all_part_clean = clean_partition  # global clean partition (NLT)
        all_true_y = nlt_true_y

        # Native
        r = run_setting(name, votes, all_true_y, nlt_emb_clean, emb_corr, all_part_clean,
                        f"native_{n_native}", n_native)
        results[f"{pool_name}_all_native"].append(r)
        # Sub4
        v4 = subsample_voters(votes, 4)
        r = run_setting(name, v4, all_true_y, nlt_emb_clean, emb_corr, all_part_clean,
                        "sub4", 4)
        results[f"{pool_name}_all_sub4"].append(r)
        # Sub7 (only if pool has >= 7)
        if n_native >= 7:
            v7 = subsample_voters(votes, 7, seed=1)
            r = run_setting(name, v7, all_true_y, nlt_emb_clean, emb_corr, all_part_clean,
                            "sub7", 7)
            results[f"{pool_name}_all_sub7"].append(r)

        # ---- CLEAN-CORRECT ----
        # Subset everything to cc_mask
        # Need partition on the subset (rebuild on subset) and embeddings on subset
        votes_cc = votes[cc_mask]
        emb_clean_cc = nlt_emb_clean[cc_mask]
        emb_corr_cc = emb_corr[cc_mask]
        true_y_cc = nlt_true_y[cc_mask]
        # Partition: rebuild on the subset (clean)
        partition_clean_cc = build_partition(emb_clean_cc, true_y_cc)

        r = run_setting(name, votes_cc, true_y_cc, emb_clean_cc, emb_corr_cc, partition_clean_cc,
                        f"native_{n_native}", n_native)
        results[f"{pool_name}_cc_native"].append(r)
        v4 = subsample_voters(votes_cc, 4)
        r = run_setting(name, v4, true_y_cc, emb_clean_cc, emb_corr_cc, partition_clean_cc,
                        "sub4", 4)
        results[f"{pool_name}_cc_sub4"].append(r)
        if n_native >= 7:
            v7 = subsample_voters(votes_cc, 7, seed=1)
            r = run_setting(name, v7, true_y_cc, emb_clean_cc, emb_corr_cc, partition_clean_cc,
                            "sub7", 7)
            results[f"{pool_name}_cc_sub7"].append(r)

        print(f"  [{i+1}/{len(SETTINGS)}] {name}: encode={enc_t:.0f}s | "
              f"native m3_gap_all={results[f'{pool_name}_all_native'][-1]['m3_gap']:.4f} "
              f"m3_gap_cc={results[f'{pool_name}_cc_native'][-1]['m3_gap']:.4f}")

        del emb_corr, emb_corr_cc, corr_images; gc.collect()

    for label, items in results.items():
        (OUT / f"{label}.json").write_text(json.dumps({"cell": label, "settings": items}, indent=2))
        print(f"  saved {label}.json ({len(items)} settings)")


# ============================================================
# Run cells A, B (Gu-arch pool) and C, D (v2 pool)
# ============================================================
def _v2_loader(name, c_dir, s_dir):
    return votes_v2(s_dir)

def _gu_arch_loader(name, c_dir, s_dir):
    return votes_gu_arch(name)


# Orchestration moved to run_stage_*.py scripts; importing this module
# only sets up CLIP, NLT, partitions, masks, and provides the runner functions.

