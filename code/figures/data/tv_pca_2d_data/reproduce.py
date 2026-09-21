"""Reproduce tv_pca_2d.pdf from mean_distributions.csv alone.

Run:
    pip install numpy scikit-learn matplotlib
    python reproduce.py
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

names, vecs, pipelines, supers = [], [], [], []
with open('mean_distributions.csv') as f:
    r = csv.DictReader(f)
    for row in r:
        names.append(row['name'])
        supers.append(row['super_family'])
        pipelines.append(row['pipeline'])
        vecs.append([float(row[f'p{i}']) for i in range(10)])
X = np.array(vecs)

pca = PCA(n_components=2)
pc = pca.fit_transform(X)
print(f'PCA explained variance ratio: {pca.explained_variance_ratio_}')

fig, ax = plt.subplots(figsize=(13, 9))
markers = {'Blur': 's', 'Noise+Geometric': '^', 'Weather': 'o', 'Digital': 'D'}
colors = {'Blur': 'tab:blue', 'Noise+Geometric': 'tab:orange',
          'Weather': 'tab:green', 'Digital': 'tab:red'}
for sf, m in markers.items():
    idx = [i for i, p in enumerate(pipelines) if p == 'CILN' and supers[i] == sf]
    if idx:
        ax.scatter(pc[idx, 0], pc[idx, 1], marker=m, s=80, color=colors[sf],
                   alpha=0.85, edgecolor='black', linewidth=0.6, label=sf)
gu = [i for i, p in enumerate(pipelines) if p == 'PL-IDN']
ax.scatter(pc[gu, 0], pc[gu, 1], marker='*', s=380, color='red',
           edgecolor='black', linewidth=1.0, label='PL-IDN')
c = [i for i, p in enumerate(pipelines) if p == 'CIFAR-10H']
ax.scatter(pc[c, 0], pc[c, 1], marker='X', s=180, color='red',
           edgecolor='black', linewidth=1.0, label='CIFAR-10H')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var.)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var.)')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tv_pca_2d_reproduced.pdf', bbox_inches='tight')
print('Saved tv_pca_2d_reproduced.pdf')
