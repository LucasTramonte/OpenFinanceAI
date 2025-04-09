# Basic - Visual Retriever Evaluation Report

## 1. Overall Performance Summary

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.3654 |
| mrr | 0.5929 |
| precision@1 | 0.5192 |
| precision@3 | 0.3269 |
| recall@1 | 0.1731 |
| recall@3 | 0.3186 |

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
| exact_match_top1 | 0.3235 |
| mrr | 0.5490 |
| precision@1 | 0.4706 |
| precision@3 | 0.2549 |
| recall@1 | 0.1569 |
| recall@3 | 0.2549 |

### Type: Short

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.4444 |
| mrr | 0.6759 |
| precision@1 | 0.6111 |
| precision@3 | 0.4630 |
| recall@1 | 0.2037 |
| recall@3 | 0.4389 |


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
| mrr | 0.7500 |
| precision@1 | 0.5000 |
| precision@3 | 0.3333 |
| recall@1 | 0.1667 |
| recall@3 | 0.3333 |

### Subject: Dividends

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.3333 |
| mrr | 0.3333 |
| precision@1 | 0.3333 |
| precision@3 | 0.1111 |
| recall@1 | 0.1111 |
| recall@3 | 0.1111 |

### Subject: Employment

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.5000 |
| mrr | 0.7500 |
| precision@1 | 0.6250 |
| precision@3 | 0.2917 |
| recall@1 | 0.2083 |
| recall@3 | 0.2917 |

### Subject: Environmental

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.5000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.8333 |
| recall@1 | 0.3333 |
| recall@3 | 0.8333 |

### Subject: Financial_Data

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.2000 |
| mrr | 0.4167 |
| precision@1 | 0.3000 |
| precision@3 | 0.2667 |
| recall@1 | 0.0917 |
| recall@3 | 0.2067 |

### Subject: Financial_Forecasts

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.3333 |
| precision@1 | 0.0000 |
| precision@3 | 0.3333 |
| recall@1 | 0.0000 |
| recall@3 | 0.3333 |

### Subject: Financial_Growth

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.7500 |
| precision@1 | 0.5000 |
| precision@3 | 0.3333 |
| recall@1 | 0.1667 |
| recall@3 | 0.3333 |

### Subject: Financial_Management

| Metric | Score |
|----------|------:|
| exact_match_top1 | 1.0000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.3333 |
| recall@1 | 0.3333 |
| recall@3 | 0.3333 |

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
| exact_match_top1 | 1.0000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.3333 |
| recall@1 | 0.3333 |
| recall@3 | 0.3333 |

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
| precision@3 | 0.5000 |
| recall@1 | 0.3333 |
| recall@3 | 0.5000 |

### Subject: Geographic

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.5000 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.5000 |
| recall@1 | 0.3333 |
| recall@3 | 0.5000 |

### Subject: Investement

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.5714 |
| mrr | 0.7143 |
| precision@1 | 0.7143 |
| precision@3 | 0.6190 |
| recall@1 | 0.2500 |
| recall@3 | 0.6429 |

### Subject: Investments

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.0000 |
| mrr | 0.3333 |
| precision@1 | 0.0000 |
| precision@3 | 0.3333 |
| recall@1 | 0.0000 |
| recall@3 | 0.3333 |

### Subject: M&A

| Metric | Score |
|----------|------:|
| exact_match_top1 | 0.6667 |
| mrr | 1.0000 |
| precision@1 | 1.0000 |
| precision@3 | 0.3333 |
| recall@1 | 0.3333 |
| recall@3 | 0.3333 |

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

### Mean Reciprocal Rank (MRR): 0.5929

✓ **Good**: The retriever often places a relevant document fairly high in the results.

### Precision@1: 0.5192

✓ **Good**: The first retrieved document is often relevant.

### Recall@3: 0.3186

❌ **Needs improvement**: The top three retrieved documents miss several relevant documents.

