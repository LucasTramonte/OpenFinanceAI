# CrossEncoder - Visual Retriever Evaluation Report

## 1. Overall Performance Summary

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.3654 |
| mrr | 0.4968 |
| precision@1 | 0.4423 |
| precision@3 | 0.2372 |
| recall@1 | 0.1391 |
| recall@3 | 0.2173 |

## 2. Embedding Similarity Metrics

| Metric | Score |
|----------|------:|
| cosine_sim_mean | 0.0913 |
| cosine_sim_median | 0.0801 |
| euclidean_dist_mean | 0.6953 |
| matching_score | 0.1087 |

### Analysis of Embedding Metrics

❌ **Low cosine similarity**: Question and corresponding image embeddings are not sufficiently similar, suggesting the model might struggle to properly align text and image.

❌ **Poor matching score**: The model struggles to associate questions with corresponding images, which may indicate an issue in the text-image relationship encoding.


## 3. Performance by Question Type

### Type: Long

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.2941 |
| mrr | 0.3922 |
| precision@1 | 0.3529 |
| precision@3 | 0.1471 |
| recall@1 | 0.1176 |
| recall@3 | 0.1471 |

### Type: Short

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.5000 |
| mrr | 0.6944 |
| precision@1 | 0.6111 |
| precision@3 | 0.4074 |
| recall@1 | 0.1796 |
| recall@3 | 0.3500 |


## 4. Performance by Question Subject

### Subject: Accounting

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |

### Subject: Business_Segments

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |

### Subject: Business_Strategy

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |

### Subject: Dividends

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.3333 |
| mrr | 0.8333 |
| precision@1 | 0.6667 |
| precision@3 | 0.3333 |
| recall@1 | 0.2222 |
| recall@3 | 0.3333 |

### Subject: Employment

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.8750 |
| mrr | 0.8750 |
| precision@1 | 0.8750 |
| precision@3 | 0.2917 |
| recall@1 | 0.2917 |
| recall@3 | 0.2917 |

### Subject: Environmental

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.5000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.3333 |
| recall@1 | 0.3333 |
| recall@3 | 0.3333 |

### Subject: Financial_Data

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.3000 |
| mrr | 0.6000 |
| precision@1 | 0.5000 |
| precision@3 | 0.3000 |
| recall@1 | 0.1317 |
| recall@3 | 0.2217 |

### Subject: Financial_Forecasts

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.5000 |
| precision@1 | 0.0000 |
| precision@3 | 0.3333 |
| recall@1 | 0.0000 |
| recall@3 | 0.3333 |

### Subject: Financial_Growth

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |

### Subject: Financial_Management

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |

### Subject: Financial_Results

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |

### Subject: Financial_Risks

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |

### Subject: Financial_Structure

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |

### Subject: General

| Metric | Score |
|----------|------:|
| exact_match_top1 | 1.0000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.3333 |
| recall@1 | 0.3333 |
| recall@3 | 0.3333 |

### Subject: Geographic

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.5000 |
| mrr | 0.7500 |
| precision@1 | 0.5000 |
| precision@3 | 0.6667 |
| recall@1 | 0.1667 |
| recall@3 | 0.6667 |

### Subject: Investement

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.5714 |
| mrr | 0.5714 |
| precision@1 | 0.5714 |
| precision@3 | 0.3810 |
| recall@1 | 0.1786 |
| recall@3 | 0.3452 |

### Subject: Investments

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |

### Subject: M&A

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.1111 |
| precision@1 | 0.0000 |
| precision@3 | 0.1111 |
| recall@1 | 0.0000 |
| recall@3 | 0.1111 |

### Subject: Regulatory

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.0000 |
| precision@1 | 0.0000 |
| precision@3 | 0.0000 |
| recall@1 | 0.0000 |
| recall@3 | 0.0000 |


## 5. Analysis and Recommendations

### Mean Reciprocal Rank (MRR): 0.4968

❌ **Needs improvement**: The retriever could better rank relevant documents.

### Precision@1: 0.4423

❌ **Needs improvement**: The first retrieved document is not sufficiently relevant.

### Recall@3: 0.2173

❌ **Needs improvement**: The top three retrieved documents miss several relevant documents.

