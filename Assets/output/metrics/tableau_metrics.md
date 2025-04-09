## Tableau des métriques Global (sans Faithfulness)

| Model            |   Borda_Global |   Faithfulness_Avg |
|:-----------------|---------------:|-------------------:|
| Answer_Qwen2     |             44 |           0.846154 |
| Answer_Qwen2.5   |             43 |           0.846154 |
| Answer_Gemma_4B  |              0 |           0.884615 |
| Answer_Gemma_12B |             13 |           0.826923 |
| Answer_PV2       |             30 |         IN PROGRESS|






## Tableau des métriques Short 

| Model            |   numerical_acc |   rouge1 |    rouge2 |   rougeL |   string_presence |
|:-----------------|----------------:|---------:|----------:|---------:|------------------:|
| Answer_Gemma_12B |        0.444444 | 0.217462 | 0.111604  | 0.201723 |          0.277778 |
| Answer_Gemma_4B  |        0.388889 | 0.154135 | 0.0697411 | 0.134767 |          0.166667 |
| Answer_PV2       |        0.611111 | 0.263643 | 0.151647  | 0.233354 |          0.388889 |
| Answer_Qwen2     |        0.555556 | 0.349403 | 0.221355  | 0.327933 |          0.388889 |
| Answer_Qwen2.5   |        0.666667 | 0.294539 | 0.173256  | 0.26821  |          0.722222 |
| Answer_PV2       |             NA  | NA | NA  | NA  |           NA|

## Tableau des métriques Long 

| Model            |     bert |   flan-t5 |   rouge1 |    rouge2 |   rougeL |
|:-----------------|---------:|----------:|---------:|----------:|---------:|
| Answer_Gemma_12B | 0.5343   | 0.0613858 | 0.253494 | 0.0602928 | 0.165765 |
| Answer_Gemma_4B  | 0.512398 | 0.0568944 | 0.230032 | 0.040173  | 0.145061 |
| Answer_PV2       | 0.577598 | 0.0820851 | 0.286641 | 0.0978303 | 0.195091 |
| Answer_Qwen2     | 0.590892 | 0.0826451 | 0.284528 | 0.0853241 | 0.198275 |
| Answer_Qwen2.5   | 0.58277  | 0.0817051 | 0.293824 | 0.0980095 | 0.204269 |
| Answer_PV2       |             NA  | NA | NA  | NA  |           NA|
