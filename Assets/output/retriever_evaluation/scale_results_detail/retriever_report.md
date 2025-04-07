# Rapport d'Évaluation du Retriever Visuel ColQwen2 - Scale à 10 pdfs

## 1. Résumé des performances globales

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.2308 |
| mrr | 0.4913 |
| precision@1 | 0.4231 |
| precision@3 | 0.2564 |
| precision@5 | 0.2000 |
| recall@1 | 0.1410 |
| recall@3 | 0.2497 |
| recall@5 | 0.3202 |

## 2. Métriques de similarité des embeddings

| Métrique | Score |
|----------|------:|
| cosine_sim_mean | 0.0566 |
| cosine_sim_median | 0.0527 |
| euclidean_dist_mean | 0.6719 |
| matching_score | 0.0870 |

### Analyse des métriques d'embedding

❌ **Faible similarité cosinus** : Les embeddings des questions et des images correspondantes ne sont pas suffisamment similaires, suggérant que le modèle pourrait avoir du mal à aligner correctement le texte et l'image.

❌ **Faible matching score** : Le modèle a du mal à associer les questions avec les images correspondantes, ce qui peut indiquer un problème dans l'encodage des relations texte-image.


## 3. Performances par type de question

### Type: Long

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.2059 |
| mrr | 0.4525 |
| precision@1 | 0.3824 |
| precision@3 | 0.1961 |
| precision@5 | 0.1529 |
| recall@1 | 0.1275 |
| recall@3 | 0.1961 |
| recall@5 | 0.2549 |

### Type: Short

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.2778 |
| mrr | 0.5648 |
| precision@1 | 0.5000 |
| precision@3 | 0.3704 |
| precision@5 | 0.2889 |
| recall@1 | 0.1667 |
| recall@3 | 0.3509 |
| recall@5 | 0.4435 |


## 4. Performances par sujet de question

### Sujet: Accounting

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.0000 |

### Sujet: Business_Segments

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.0000 |

### Sujet: Business_Strategy

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.6250 |
| precision@1 | 0.5000 |
| precision@3 | 0.1667 |
| precision@5 | 0.2000 |
| recall@1 | 0.1667 |
| recall@3 | 0.1667 |
| recall@5 | 0.3333 |

### Sujet: Dividends

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.1333 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.1333 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.2222 |

### Sujet: Employment

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.6250 |
| mrr | 0.8542 |
| precision@1 | 0.7500 |
| precision@3 | 0.3333 |
| precision@5 | 0.2000 |
| recall@1 | 0.2500 |
| recall@3 | 0.3333 |
| recall@5 | 0.3333 |

### Sujet: Environmental

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.5000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.8333 |
| precision@5 | 0.6000 |
| recall@1 | 0.3333 |
| recall@3 | 0.8333 |
| recall@5 | 1.0000 |

### Sujet: Financial_Data

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.1000 |
| mrr | 0.3167 |
| precision@1 | 0.2000 |
| precision@3 | 0.2000 |
| precision@5 | 0.1800 |
| recall@1 | 0.0583 |
| recall@3 | 0.1483 |
| recall@5 | 0.2233 |

### Sujet: Financial_Forecasts

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.2000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.2000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.3333 |

### Sujet: Financial_Growth

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.5000 |
| precision@1 | 0.5000 |
| precision@3 | 0.1667 |
| precision@5 | 0.1000 |
| recall@1 | 0.1667 |
| recall@3 | 0.1667 |
| recall@5 | 0.1667 |

### Sujet: Financial_Management

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.0000 |

### Sujet: Financial_Results

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.0000 |

### Sujet: Financial_Risks

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.0000 |

### Sujet: Financial_Structure

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.0000 |

### Sujet: General

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 1.0000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.5000 |
| precision@5 | 0.4000 |
| recall@1 | 0.3333 |
| recall@3 | 0.5000 |
| recall@5 | 0.6667 |

### Sujet: Geographic

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.6667 |
| precision@5 | 0.4000 |
| recall@1 | 0.3333 |
| recall@3 | 0.6667 |
| recall@5 | 0.6667 |

### Sujet: Investement

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.4286 |
| mrr | 0.5714 |
| precision@1 | 0.5714 |
| precision@3 | 0.4286 |
| precision@5 | 0.3143 |
| recall@1 | 0.2024 |
| recall@3 | 0.4524 |
| recall@5 | 0.5357 |

### Sujet: Investments

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.3333 |
| precision@5 | 0.2000 |
| recall@1 | 0.3333 |
| recall@3 | 0.3333 |
| recall@5 | 0.3333 |

### Sujet: M&A

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.5000 |
| precision@1 | 0.3333 |
| precision@3 | 0.2222 |
| precision@5 | 0.1333 |
| recall@1 | 0.1111 |
| recall@3 | 0.2222 |
| recall@5 | 0.2222 |

### Sujet: Regulatory

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0667 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.0667 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.1111 |


## 5. Analyse et recommandations

### Mean Reciprocal Rank (MRR): 0.4913

❌ **À améliorer**: Le retriever pourrait mieux classer les documents pertinents.

### Precision@1: 0.4231

❌ **À améliorer**: Le premier document récupéré n'est pas suffisamment pertinent.

### Recall@3: 0.2497

❌ **À améliorer**: Les trois premiers documents récupérés manquent plusieurs documents pertinents.

