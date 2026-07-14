# OCR Model Evaluation Script

A tool for evaluating OCR models on structured information extraction tasks. Calls models through an OpenAI-compatible API and computes accuracy by dataset and by xfund language.

## Requirements

```bash
pip install openai json_repair
```

## Usage

```bash
python evaluate/evaluate_ocr_openai.py \
    --jsonl /path/to/test.jsonl \
    --base_url http://localhost:8111/v1 \
    --model paddleocr \
    --image_root /path/to/data/root \
    --output_dir output
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--jsonl` | Yes | - | Path to the JSONL evaluation file |
| `--base_url` | Yes | - | OpenAI-compatible API endpoint |
| `--model` | No | `paddleocr` | Prompt type: `paddleocr`, `hunyuan`, or `glm` |
| `--api_key` | No | `EMPTY` | API key for authentication |
| `--image_root` | No | - | Root directory for resolving relative image paths |
| `--output_dir` | No | `output` | Directory for output files |

## JSONL Data Format

Each line is a JSON object:

```json
{
  "image_info": [{"image_url": "wildreceipt/image_files/xxx.jpg"}],
  "text_info": [
    {"text": "OCR:{...}", "tag": "mask"},
    {"text": "{\"Store name\": \"...\", \"Items\": [{\"item\": \"...\"}]}", "tag": "no_mask"}
  ]
}
```

- `image_url` is typically a relative path; use `--image_root` to resolve it.
- `text_info` with `tag: "no_mask"` is the ground truth JSON.
- `text_info` with `tag: "mask"` is overwritten at runtime by the prompt constructed from ground truth.

## Prompt Construction

Each `--model` option constructs the prompt differently from the ground truth JSON.

### `paddleocr` - PaddleOCR-VL

Preserves the full nested JSON structure. Values are replaced with empty strings. For lists containing dicts, the first element's key structure is kept.

**Example** ground truth:
```json
{"Store name": "RadHouse", "Total": "238.50", "Items": [{"item": "Coconut", "price": "2.50"}]}
```

**Prompt:**
```
OCR:{"Store name": "", "Total": "", "Items": [{"item": "", "price": ""}]}
```

The model is expected to return a JSON with the same structure, filling in values from the image.

### `hunyuan` - HunyuanOCR

Extracts only the top-level keys and embeds them in a natural language sentence.

**Prompt:**
```
从图片中提取字段内容: ['Store name', 'Total', 'Items'] 并以 JSON 格式返回。
```

### `glm` - GLM-OCR

Preserves the nested structure like `paddleocr`, but uses empty strings for all values including lists (no list element expansion).

**Prompt:**
```
请按下列JSON格式输出图中信息:{"Store name": "", "Total": "", "Items": ""}
```

## Output

Three files are saved to `--output_dir`:

| File | Content |
|------|---------|
| `evaluation_results.json` | Full evaluation results with per-sample details and aggregate statistics |
| `predictions.jsonl` | Per-sample predictions for external analysis |
| `evaluation.log` | Processing log with per-sample accuracy |

### Statistics in `evaluation_results.json`

- **Overall** - total/processed/failed samples, average accuracy, total time, avg time per sample
- **By dataset** - FATURA, SROIE, wildreceipt, CORD-v2, xfund, nutritional, SIBR
- **By xfund language** - de, es, fr, it, ja, pt, zh (only for xfund samples)

### Console Output

```
======================================================================
EVALUATION SUMMARY
======================================================================
Model: PaddleOCR-VL (OpenAI API)
Prompt Type: paddleocr
Total Samples: 889
Processed Samples: 889
Failed Samples: 0
Overall Accuracy: 75.99%
Total Time: 1234.56s
Avg Time/Sample: 1.389s

----------------------------------------------------------------------
STATS BY DATASET
----------------------------------------------------------------------
Dataset              Total  Processed   Failed   Accuracy
----------------------------------------------------------------------
CORD-v2                84         84        0    75.00%
FATURA                255        255        0    95.87%
SIBR                  ...        ...      ...      ...
SROIE                 117        117        0    87.82%
nutritional            35         35        0    63.72%
wildreceipt           189        189        0    64.81%
xfund                 125        125        0    47.82%

----------------------------------------------------------------------
STATS BY XFUND LANGUAGE
----------------------------------------------------------------------
Language       Total  Processed   Failed   Accuracy
----------------------------------------------------------------------
de                24         24        0    49.76%
es                ...        ...      ...      ...
zh                18         18        0    50.10%
======================================================================
```
