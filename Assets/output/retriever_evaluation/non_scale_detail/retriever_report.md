# Rapport d'Évaluation du Retriever Visuel ColQwen2

## 1. Résumé des performances globales

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.3462 |
| mrr | 0.5814 |
| precision@1 | 0.4808 |
| precision@3 | 0.3205 |
| precision@5 | 0.2308 |
| recall@1 | 0.1603 |
| recall@3 | 0.3138 |
| recall@5 | 0.3699 |

## 2. Métriques de similarité des embeddings

| Métrique | Score |
|----------|------:|
| cosine_sim_mean | 0.0481 |
| cosine_sim_median | 0.0552 |
| euclidean_dist_mean | 0.7031 |
| matching_score | 0.1087 |

### Analyse des métriques d'embedding

❌ **Faible similarité cosinus** : Les embeddings des questions et des images correspondantes ne sont pas suffisamment similaires, suggérant que le modèle pourrait avoir du mal à aligner correctement le texte et l'image.

❌ **Faible matching score** : Le modèle a du mal à associer les questions avec les images correspondantes, ce qui peut indiquer un problème dans l'encodage des relations texte-image.


## 3. Performances par type de question

### Type: Long

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.2941 |
| mrr | 0.5255 |
| precision@1 | 0.4118 |
| precision@3 | 0.2451 |
| precision@5 | 0.1706 |
| recall@1 | 0.1373 |
| recall@3 | 0.2451 |
| recall@5 | 0.2843 |

### Type: Short

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.4444 |
| mrr | 0.6870 |
| precision@1 | 0.6111 |
| precision@3 | 0.4630 |
| precision@5 | 0.3444 |
| recall@1 | 0.2037 |
| recall@3 | 0.4435 |
| recall@5 | 0.5315 |


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
| mrr | 0.2000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| precision@5 | 0.2000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |
| recall@5 | 0.3333 |

### Sujet: Business_Strategy

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.7500 |
| precision@1 | 0.5000 |
| precision@3 | 0.3333 |
| precision@5 | 0.2000 |
| recall@1 | 0.1667 |
| recall@3 | 0.3333 |
| recall@5 | 0.3333 |

### Sujet: Dividends

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.3333 |
| mrr | 0.3333 |
| precision@1 | 0.3333 |
| precision@3 | 0.1111 |
| precision@5 | 0.0667 |
| recall@1 | 0.1111 |
| recall@3 | 0.1111 |
| recall@5 | 0.1111 |

### Sujet: Employment

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.5000 |
| mrr | 0.7812 |
| precision@1 | 0.6250 |
| precision@3 | 0.2917 |
| precision@5 | 0.2000 |
| recall@1 | 0.2083 |
| recall@3 | 0.2917 |
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
| exact_match_top1 | 0.2000 |
| mrr | 0.4367 |
| precision@1 | 0.3000 |
| precision@3 | 0.2667 |
| precision@5 | 0.2400 |
| recall@1 | 0.0917 |
| recall@3 | 0.2150 |
| recall@5 | 0.3150 |

### Sujet: Financial_Forecasts

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.2500 |
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
| mrr | 0.7500 |
| precision@1 | 0.5000 |
| precision@3 | 0.3333 |
| precision@5 | 0.2000 |
| recall@1 | 0.1667 |
| recall@3 | 0.3333 |
| recall@5 | 0.3333 |

### Sujet: Financial_Management

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.3333 |
| precision@1 | 0.0000 |
| precision@3 | 0.3333 |
| precision@5 | 0.2000 |
| recall@1 | 0.0000 |
| recall@3 | 0.3333 |
| recall@5 | 0.3333 |

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
| exact_match_top1 | 1.0000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.3333 |
| precision@5 | 0.2000 |
| recall@1 | 0.3333 |
| recall@3 | 0.3333 |
| recall@5 | 0.3333 |

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
| precision@5 | 0.3000 |
| recall@1 | 0.3333 |
| recall@3 | 0.5000 |
| recall@5 | 0.5000 |

### Sujet: Geographic

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.5000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.5000 |
| precision@5 | 0.4000 |
| recall@1 | 0.3333 |
| recall@3 | 0.5000 |
| recall@5 | 0.6667 |

### Sujet: Investement

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.5714 |
| mrr | 0.7143 |
| precision@1 | 0.7143 |
| precision@3 | 0.6190 |
| precision@5 | 0.4000 |
| recall@1 | 0.2500 |
| recall@3 | 0.6429 |
| recall@5 | 0.6786 |

### Sujet: Investments

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.5000 |
| precision@1 | 0.0000 |
| precision@3 | 0.3333 |
| precision@5 | 0.2000 |
| recall@1 | 0.0000 |
| recall@3 | 0.3333 |
| recall@5 | 0.3333 |

### Sujet: M&A

| Métrique | Score |
|----------|------:|
| exact_match_top1 | 0.6667 |
| mrr | 0.7778 |
| precision@1 | 0.6667 |
| precision@3 | 0.3333 |
| precision@5 | 0.2000 |
| recall@1 | 0.2222 |
| recall@3 | 0.3333 |
| recall@5 | 0.3333 |

### Sujet: Regulatory

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


## 5. Analyse et recommandations

### Mean Reciprocal Rank (MRR): 0.5814

✓ **Bon**: Le retriever place souvent un document pertinent assez haut dans les résultats.

### Precision@1: 0.4808

❌ **À améliorer**: Le premier document récupéré n'est pas suffisamment pertinent.

### Recall@3: 0.3138

❌ **À améliorer**: Les trois premiers documents récupérés manquent plusieurs documents pertinents.

