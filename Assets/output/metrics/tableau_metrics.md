## Tableau des métriques Global

| Model                     |   Borda_Global | Faithfulness_Avg |
|:--------------------------|---------------:|------------------|
| Answer_Qwen2              |             60 | 84.62%           |
| Answer_Qwen2.5            |             58 | 84.62%           |
| Answer_Gemma_4B           |             21 | 88.46%           |
| Answer_Gemma_12B          |             22 | 82.69%           |
| Answer_PV2                |             43 | 78.43%           |
| Answer_langchain_pipeline |              6 | 65.38%           |

## Tableau des métriques Short

| Model                     | faithfulness | numerical_acc | rouge1  | rouge2  | rougeL  | string_presence |
|:--------------------------|:-------------|:--------------|:--------|:--------|:--------|:---------------|
| Answer_Gemma_12B          | 66.67%       | 44.44%        | 21.75%  | 11.16%  | 20.17%  | 27.78%         |
| Answer_Gemma_4B           | 77.78%       | 50.00%        | 23.68%  | 11.94%  | 21.74%  | 33.33%         |
| Answer_PV2                | 82.35%       | 61.11%        | 26.36%  | 15.16%  | 23.34%  | 38.89%         |
| Answer_Qwen2              | 77.78%       | 55.56%        | 34.94%  | 22.14%  | 32.79%  | 38.89%         |
| Answer_Qwen2.5            | 88.89%       | 66.67%        | 29.45%  | 17.33%  | 26.82%  | 72.22%         |
| Answer_langchain_pipeline | 77.78%       | 44.44%        | 19.64%  | 7.87%   | 18.21%  | 22.22%         |

## Tableau des métriques Long

| Model                     | bert    | faithfulness | flan-t5 | rouge1  | rouge2  | rougeL  |
|:--------------------------|:--------|:-------------|:--------|:--------|:--------|:--------|
| Answer_Gemma_12B          | 53.43%  | 91.18%       | 6.13%   | 25.35%  | 6.03%   | 16.58%  |
| Answer_Gemma_4B           | 51.24%  | 94.12%       | 5.69%   | 23.00%  | 4.02%   | 14.51%  |
| Answer_PV2                | 57.76%  | 76.47%       | 8.21%   | 28.66%  | 9.78%   | 19.51%  |
| Answer_Qwen2              | 59.09%  | 88.24%       | 8.26%   | 28.45%  | 8.53%   | 19.83%  |
| Answer_Qwen2.5            | 58.28%  | 82.35%       | 8.17%   | 29.38%  | 9.80%   | 20.43%  |
| Answer_langchain_pipeline | 51.65%  | 58.82%       | 8.92%   | 18.42%  | 3.65%   | 12.28%  |