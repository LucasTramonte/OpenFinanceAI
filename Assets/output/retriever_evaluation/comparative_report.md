# Rapport Comparatif d'Évaluation du Retriever Visuel ColQwen2

Ce rapport compare les performances du Retriever Visuel ColQwen2 dans deux configurations:
- **Standard**: Configuration de base
- **Scale PDFs**: Configuration avec scaling à 10 PDFs. Initialement , nous avons effectué avec 5 PDFs et cela a donné des résultats identiques à ceux obtenus avec 10 PDFs, ce qui suggère que les dégradations de performances apparaissent dès l'augmentation du corpus à 5 PDFs et ne s'aggravent pas significativement au-delà.

## 1. Comparaison des performances globales

| Métrique | Standard | Scale à 10 PDFs | Différence |
|----------|--------:|----------------:|-----------:|
| exact_match_top1 | 0.3462 | 0.2308 | -0.1154 🔻 |
| mrr | 0.5814 | 0.4913 | -0.0901 🔻 |
| precision@1 | 0.4808 | 0.4231 | -0.0577 🔻 |
| precision@3 | 0.3205 | 0.2564 | -0.0641 🔻 |
| precision@5 | 0.2308 | 0.2000 | -0.0308 🔻 |
| recall@1 | 0.1603 | 0.1410 | -0.0193 🔻 |
| recall@3 | 0.3138 | 0.2497 | -0.0641 🔻 |
| recall@5 | 0.3699 | 0.3202 | -0.0497 🔻 |

**Interprétation**: On observe une dégradation systématique des performances avec le scaling à 10 PDFs. La précision des correspondances exactes (exact_match_top1) diminue de 11.5 points, suggérant que l'augmentation du corpus rend plus difficile la récupération précise des documents pertinents.

## 2. Comparaison des métriques de similarité des embeddings

| Métrique | Standard | Scale à 10 PDFs | Différence |
|----------|--------:|----------------:|-----------:|
| cosine_sim_mean | 0.0481 | 0.0566 | +0.0085 🔼 |
| cosine_sim_median | 0.0552 | 0.0527 | -0.0025 🔻 |
| euclidean_dist_mean | 0.7031 | 0.6719 | -0.0312 🔼 |
| matching_score | 0.1087 | 0.0870 | -0.0217 🔻 |

**Interprétation**: Les métriques de similarité montrent des changements mitigés. La similarité cosinus moyenne est légèrement meilleure dans la version scale, mais le matching score est inférieur, indiquant que le modèle a plus de difficulté à associer correctement les questions aux images correspondantes avec l'augmentation du corpus.

## 3. Comparaison des performances par type de question

### Type: Long

| Métrique | Standard | Scale à 10 PDFs | Différence |
|----------|--------:|----------------:|-----------:|
| exact_match_top1 | 0.2941 | 0.2059 | -0.0882 🔻 |
| mrr | 0.5255 | 0.4525 | -0.0730 🔻 |
| precision@1 | 0.4118 | 0.3824 | -0.0294 🔻 |
| precision@3 | 0.2451 | 0.1961 | -0.0490 🔻 |
| precision@5 | 0.1706 | 0.1529 | -0.0177 🔻 |
| recall@1 | 0.1373 | 0.1275 | -0.0098 🔻 |
| recall@3 | 0.2451 | 0.1961 | -0.0490 🔻 |
| recall@5 | 0.2843 | 0.2549 | -0.0294 🔻 |

### Type: Short

| Métrique | Standard | Scale à 10 PDFs | Différence |
|----------|--------:|----------------:|-----------:|
| exact_match_top1 | 0.4444 | 0.2778 | -0.1666 🔻 |
| mrr | 0.6870 | 0.5648 | -0.1222 🔻 |
| precision@1 | 0.6111 | 0.5000 | -0.1111 🔻 |
| precision@3 | 0.4630 | 0.3704 | -0.0926 🔻 |
| precision@5 | 0.3444 | 0.2889 | -0.0555 🔻 |
| recall@1 | 0.2037 | 0.1667 | -0.0370 🔻 |
| recall@3 | 0.4435 | 0.3509 | -0.0926 🔻 |
| recall@5 | 0.5315 | 0.4435 | -0.0880 🔻 |

**Interprétation**: Les questions courtes semblent plus affectées par le scaling que les questions longues, avec une baisse notable de 16.7 points sur exact_match_top1. Cela suggère que les requêtes courtes sont plus ambiguës lorsque le corpus augmente, tandis que les questions longues, qui contiennent plus de contexte, résistent mieux à l'élargissement du corpus.

## 4. Principaux changements par sujet de question

| Sujet | Métrique | Standard | Scale à 10 PDFs | Différence |
|-------|----------|--------:|----------------:|-----------:|
| Employment | exact_match_top1 | 0.5000 | 0.6250 | +0.1250 🔼 |
| Investement | exact_match_top1 | 0.5714 | 0.4286 | -0.1428 🔻 |
| M&A | exact_match_top1 | 0.6667 | 0.0000 | -0.6667 🔻 |
| Financial_Data | mrr | 0.4367 | 0.3167 | -0.1200 🔻 |
| Environmental | mrr | 1.0000 | 1.0000 | 0.0000 ⟹ |

**Interprétation**: Les performances varient considérablement selon les sujets. Le sujet "Employment" montre une amélioration avec le scaling, tandis que "M&A" présente une dégradation drastique. Certains sujets comme "Environmental" maintiennent leurs performances quelle que soit la taille du corpus.

On remarque que passer à un corpus plus large impacte différemment les différents domaines de questions, ce qui suggère que certains sujets sont plus sensibles à la dilution d'information que d'autres. 

Les performances varient considérablement selon les sujets. Le sujet "Employment" montre une amélioration avec le scaling, tandis que "M&A" présente une dégradation drastique. Certains sujets comme "Environmental" maintiennent leurs performances quelle que soit la taille du corpus.