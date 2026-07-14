
# PaddleOCR-VL-KIE Fine-tuning on KIE Datasets for Information Extraction

> 💥💥💥 [PaddleOCR-VL-XFUND](https://aistudio.baidu.com/projectdetail/10253559) **Pro** **Max** version models & OCR-KIE dataset
>
> **Pro**
>
> - Base model upgraded from PaddleOCR-VL-1.5 to PaddleOCR-VL-1.6
>
> - LLM information extraction → `RLHF` information extraction script
>
> - Proposed a 'Character Pixel Density-based Image Scaling Algorithm'
>
> **Max**
>
> - Integrated 7 datasets across 4 categories
>
> - Generalization testing on 3 datasets
>
> ---

![](https://ai-studio-static-online.cdn.bcebos.com/2afba10ecddd47a49a6ec93e542ce8e30c7079fb0af7458e9fb91cb8152329c1?v=1)


![](https://ai-studio-static-online.cdn.bcebos.com/c20368be09c649b0a7a6f6d5637b38bb54224ea47d334f7e80e91dcc53b63333?v=1)

> Note:
>
> In the figures above, PaddleOCR-VL-KIE refers to the fine-tuned model based on PaddleOCR-VL-1.6; PaddleOCR-VL-KIE-1.5 is based on PaddleOCR-VL-1.5, used for comparing the two base models.
>
> GLM-OCR defaults to using the Q8 quantized mmproj model; GLM-OCR F16 uses the F16 mmproj model.
>

![](https://ai-studio-static-online.cdn.bcebos.com/3bf2f669b0264bceba4f098f2f0127be38824fe52ac547f2be33488f9618978c?v=1)

---

## ✨️ Introduction

> The model can be downloaded: https://modelscope.cn/models/megemini/PaddleOCR-VL-KIE/summary

The [PaddleOCR-VL-XFUND](https://aistudio.baidu.com/projectdetail/10253559) article used data from [XFUND: A Multilingual Form Understanding Benchmark](https://github.com/doc-analysis/XFUND) for fine-tuning to achieve key information extraction from forms. However, models fine-tuned on a single dataset inevitably exhibit recognition bias and insufficient generalization.

This article introduces the **OCR-KIE** dataset, which collects and consolidates **7** datasets from **4** domains currently available in the open-source community. Based on the **PaddleOCR-VL-1.6** model, we fine-tuned to obtain the **PaddleOCR-VL-KIE** model, and validated its generalization capability on **3** unseen datasets.

Results show:

- PaddleOCR-VL-KIE leads the second place HunyuanOCR by a large margin with 75.99% vs 45.06%
- PaddleOCR-VL-KIE's generalization capability achieves 66.83%, comparable to GLM-OCR's 67.59%

> Note:
>
> As will be seen in this article, the model's recognition performance on the XFUND dataset differs significantly from [PaddleOCR-VL-XFUND](https://aistudio.baidu.com/projectdetail/10253559).
>
> This is because the OCR-KIE dataset applied certain processing to XFUND and other datasets, such as image scaling, and did not follow XFUND's original train/val split to prevent models from achieving inflated scores by training on the validation and test sets.
>
> Therefore, the test scores in this article are only meaningful for the OCR-KIE dataset and should not be compared with evaluation results from the original datasets.
>

---

## ✨️ Data Preparation: OCR-KIE

This article introduces the **OCR-KIE** dataset.

### 💫 Dataset Information

The original data information is as follows:

| Dataset | Category | Description | Records |
|---------|----------|-------------|---------|
| FATURA | Invoices | Invoice documents (50 templates × 50 instances) | 2,500 |
| SROIE | Receipts | Receipt information extraction | 973 |
| wildreceipt | Receipts | Wild receipt recognition | 1,739 |
| CORD-v2 | Receipts | Commercial receipts | 1,000 |
| xfund | Forms | Multi-language form understanding (7 languages) | 1,393 |
| SIBR | Forms | Structured information in documents | 1,000 |
| nutritional-data-poie-1 | Labels | Nutritional information labels | 483 |

Classified by image category into: Invoices, Receipts, Forms, and Labels.

![](https://ai-studio-static-online.cdn.bcebos.com/03cd12faa9f94b95b6c1d89d2f6f83d933ee2a2d717248f494c3de5cded14c84?v=1)

The following processing was applied to each dataset:

1. Using annotations from the original datasets, scripts were written via the RLHF method to uniformly organize annotations into the JSON format required for fine-tuning
2. The FATURA original dataset has 50 templates with 200 records each; to ensure data balance across the overall dataset, only 50 records per template were extracted, totaling 2,500 records
3. Converted PNG format images to smaller JPG format
4. Applied the 'Character Pixel Density-based Image Scaling Algorithm' and SWT algorithm mentioned later in this article to proportionally scale down images, reducing VRAM usage
5. Translated the `header` field based on different languages to replace `header`
6. Mixed original train, valid, and test data, then re-split into train and test sets at a 9:1 ratio
7. Extended the dataset by extracting partial fields; the extended training set contains a total of 48,000 records
8. Removed the `other` field which has little recognition significance

> Translation rules for the `header` field:

| Language | Translation |
|----------|-------------|
| Chinese (zh) | `标题` |
| German (de) | `Kopfzeile` |
| Spanish (es) | `Encabezado` |
| French (fr) | `En-tête` |
| Italian (it) | `Intestazione` |
| Japanese (ja) | `ヘッダー` |
| Portuguese (pt) | `Cabeçalho` |

The final format for each record is:

```json
{
  "image_info": [
    {"matched_text_index": 0, "image_url": "path/to/image.jpg"}
  ],
  "text_info": [
    {"text": "OCR:{}", "tag": "mask"},
    {"text": "{\"key\": \"value\", ...}", "tag": "no_mask"}
  ]
}
```

- `image_info`: relative path to the image
- `text_info[0]`: prompt for recognition
- `text_info[1]`: fields to be recognized

The recognition prompts are:

- Original records: `OCR:{}` with all fields in no_mask
- Extended records: `OCR:{"field1": "", "field2": []}` with only selected fields
- Structured mask records: preserves full structure with type-aware placeholders

Field design:

- String value: `"field": ""`
- List of strings: `"field": []`
- List of dicts: `"field": [{"key": ""}]` (preserves the structure of the first element)
- Nested dict: `"field": {"k1": "", "k2": ""}`

For comparison, here are the prompt designs of GLM-OCR and HunyuanOCR:

GLM-OCR:

- 请按下列JSON格式输出图中信息:{key:"", ...}

HunyuanOCR:

- 从图片中提取字段内容: [key1, key2, ...] 并以 JSON 格式返回。

The key differences of OCR-KIE's prompt design compared to both:

- OCR-KIE supports full information extraction, meaning no key specification is needed
- OCR-KIE supports list extraction
- OCR-KIE's prompt is more concise and inherits from PaddleOCR-VL's `OCR:` prompt

### 💫 Information Extraction Script

Original datasets, such as XFUND, have annotations not formatted for VLM. Here is an XFUND example:

```json
{
    "height": 3508, # image height
    "width": 2480,  # image width
    "ocr_info": [
        {
            "text": "邮政地址:",  # single text content
            "label": "question", # text category
            "bbox": [261, 802, 483, 859], # text bounding box
            "id": 54,  # text index
            "linking": [[54, 60]], # relationship between current text and other texts [question, answer]
            "words": []
        },
        {
            "text": "湖南省怀化市市辖区",
            "label": "answer",
            "bbox": [487, 810, 862, 859],
            "id": 60,
            "linking": [[54, 60]],
            "words": []
        }
    ]
}
```

Therefore, the annotation data from these 7 datasets needs to be organized into the structure required for fine-tuning.

[PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) specifies the data format requirements for fine-tuning PaddleOCR-VL:

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./assets/table_example.jps"},
    ],
    "text_info": [
        {"text": "OCR:", "tag": "mask"},
        {"text": "দডর মথ বধ বকসট একনজর দখই চনত পরল তর অনমন\nঠক পনতই লকয রখছ\nর নচ থকই চচয বলল কশর, "এইই; পযছ! পযছ!'\nওপর", "tag": "no_mask"},
    ]
}
```

As proposed in [PaddleOCR-VL-XFUND](https://aistudio.baidu.com/projectdetail/10253559), LLMs can be used to help organize annotation information. The basic steps are:

- Construct KV pairs from the link field in XFUND
- Use an LLM to consolidate KV pairs into meaningful JSON data
- Build the ERNIEKit SFT VL Dataset Format

However, datasets constructed this way suffer from the following issues due to differences in the LLM used:

- Model hallucination leading to incorrect results
- Inconsistent results across different models, making reproduction difficult
- Slow processing speed and high resource consumption

Subsequently, attempts were made using pipeline approaches with multiple LLMs in series, scoring supervision methods, etc., but these issues could never be fully resolved.

This article adopts the RLHF `LLM-Human Feedback-Script Generation` approach for data annotation:

1. Use an LLM to generate an annotation script based on the original dataset's annotations
2. Use this script to generate annotations, then have both the LLM and humans jointly review the annotation quality
3. Based on feedback, use the LLM to modify the script and add annotation rules
4. Repeat step 2 until the script meets requirements

This approach produces reproducible annotations with high execution efficiency, and rules can be continuously added to handle specific cases.

![](https://ai-studio-static-online.cdn.bcebos.com/bc2b8aef8c8c4410a998541fddbfb446d82bf352991241219c27f203ab662b28?v=1)

### 💫 Character Pixel Density-based Image Scaling Algorithm

This algorithm is based on the following assumption:

> **In optical character recognition (OCR) tasks, as long as the number of pixels occupied by each character exceeds a certain threshold, it will no longer have a substantial impact on the recognition performance.**

In other words, each character only needs a minimum number of pixels to ensure recognition; going far beyond that yields diminishing returns.

Some images in the original datasets have very high resolution; for example, XFUND images are generally above `2000 × 3000`.

Such large images consume significant VRAM for both training and inference, but the images themselves do not actually need such high resolution to carry their text information.

Therefore, the goal is to reduce image dimensions without losing important text information.

Currently, there is no well-established method for this. The similar SWT (Stroke Width Transform) algorithm can identify text by computing stroke width.

However, this algorithm targets ordinary life scenes rather than document images, and practical testing shows it cannot effectively scale down document images.

This article proposes a Character Pixel Density-based Image Scaling Algorithm for document and text images:

1. Use Otsu's threshold to binarize the grayscale image, separating text from background (typically white)
2. Count the number of black pixels in the binarized image (text is typically black)
3. Divide the character count from annotations by the pixel count to obtain `pixels per character`
4. Since images contain not only text but also lines, noise, etc., apply a margin discount to the above value to get `effective pixels per character`
5. Calculate the image scaling ratio based on the preset `minimum pixels per character` and `effective pixels per character`
6. Set maximum scaling ratio, maximum image size, etc. as algorithm fallbacks

![](https://ai-studio-static-online.cdn.bcebos.com/91f2278c12c445b19165dd9185531342fe6168562419434cac459aa5322dca5f?v=1)


Below is the core algorithm code:

```python
def algorithm_density(image, char_count=None, min_pixels_per_char=50,
                      margin_ratio=1.5, dark_text=True, min_scale=0.4,
                      bboxes=None, min_char_width=3, min_char_height=8):
    """
    Scale image based on text pixel density.

    Core idea (user's algorithm):
      1. Binarise and count foreground (text) pixels
      2. Compute average pixels per character from ground-truth char count
      3. Apply a margin to account for non-text foreground noise
      4. scale = sqrt(min_pixels_per_char / effective_pixels_per_char)
      5. Clamp with min_scale and bbox floor

    Returns:
        (scaled_image, scale, info_dict)
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarise (assume dark text on light background)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if not dark_text:
        binary = cv2.bitwise_not(binary)

    text_pixels = int(np.sum(binary == 255))
    total_pixels = h * w
    text_ratio = text_pixels / total_pixels

    # Estimate char count from connected components when GT is not available
    if char_count is None or char_count <= 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)
                 if stats[i, cv2.CC_STAT_AREA] > 20]
        if areas:
            avg_area = np.mean(areas)
            if avg_area > 400:
                char_count = max(1, int(sum(areas) / 200))
            else:
                char_count = max(1, len(areas))
        else:
            char_count = 1

    # Compute density-based scale
    avg_pixels_per_char = text_pixels / char_count if char_count > 0 else total_pixels

    # Dynamic margin: denser text = more aggressive scaling
    if text_ratio > 0.3:
        dynamic_margin = margin_ratio * 0.8
    elif text_ratio > 0.1:
        dynamic_margin = margin_ratio * 1.0
    else:
        dynamic_margin = margin_ratio * 1.4

    effective_pixels_per_char = avg_pixels_per_char / dynamic_margin
    scale_density = 1.0
    if effective_pixels_per_char > 0:
        scale_density = math.sqrt(min_pixels_per_char / effective_pixels_per_char)

    # Bbox floor: protect very thin characters (punctuation etc.)
    scale_floor = 0.0
    if bboxes and len(bboxes) > 0:
        min_bbox_w = min(b[2] - b[0] for b in bboxes if b[2] > b[0])
        min_bbox_h = min(b[3] - b[1] for b in bboxes if b[3] > b[1])
        need_protect_w = min_bbox_w < min_char_width * 2
        need_protect_h = min_bbox_h < min_char_height * 2
        floor_w = (1.0 / min_bbox_w) if (min_bbox_w > 0 and need_protect_w) else 0.0
        floor_h = (min_char_height / 2 / min_bbox_h) if (min_bbox_h > 0 and need_protect_h) else 0.0
        scale_floor = max(floor_w, floor_h)

    # Fuse
    scale = max(scale_density, scale_floor)
    scale = max(scale, min_scale)
    scale = min(scale, 1.0)

    # Round to multiple of 32
    new_w = max(32, int(w * scale))
    new_h = max(32, int(h * scale))
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32
    scale_actual = new_w / w

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    info = {
        "algorithm": "density",
        "scale": round(scale_actual, 4),
        "new_size": (new_w, new_h),
        "text_pixels": text_pixels,
        "char_count": char_count,
        "avg_pixels_per_char": round(avg_pixels_per_char, 2),
        "text_ratio": round(text_ratio, 4),
    }
    return resized, scale_actual, info
```

Below are the results on the xFund dataset (7 languages, 1393 images):

| Language | Median scale | Typical output | Avg reduction |
|----------|-------------|----------------|--------------|
| zh | 0.400 | 992×1376 | 84% |
| de | 0.581 | 1440×2048 | 66% |
| es | 0.581 | 1440×2048 | 66% |
| fr | 0.581 | 1440×2048 | 66% |
| it | 0.581 | 1440×2048 | 66% |
| ja | 0.400 | 992×1376 | 84% |
| pt | 0.581 | 1440×2048 | 66% |

---

## ✨️ Model Fine-tuning

The fine-tuning process can be referenced in [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md).

First, install ERNIE:

```bash
cd work
git clone https://gitee.com/PaddlePaddle/ERNIE
cd ERNIE
python -m pip install -r requirements/gpu/requirements.txt
python -m pip install -e .
python -m pip install tensorboard
python -m pip install opencv-python-headless
python -m pip install numpy==1.26.4
```

Then, modify the configuration file and copy it to overwrite the original configuration:

```bash
cp work/sft_config/run_ocr_vl_sft_16k.yaml \
  work/ERNIE/examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

Download the PaddleOCR-VL-1.6 model using aistudio:

```bash
aistudio download --model PaddlePaddle/PaddleOCR-VL-1.6 --local_dir PaddleOCR-VL-1.6
```

Finally, execute the fine-tuning command. Fine-tuning was performed on AI Studio's A100 environment.

> V100 environment cannot run fine-tuning but can perform model inference.

```bash
cd work/ERNIE; CUDA_VISIBLE_DEVICES=0 python -m erniekit.launcher train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

Training logs:

![](https://ai-studio-static-online.cdn.bcebos.com/36e4b973664d447a8fd7863055604aeac7e401b8754347a3b9c41ee535abe37b?v=1)

1. PaddleOCR-VL-1.5/1.6 fine-tuning loss is essentially consistent
2. PaddleOCR-VL-XFUND has a lower initial loss because it was fine-tuned from the PaddleOCR-VL-Receipt model
3. PaddleOCR-VL-XFUND's final loss is higher than PaddleOCR-VL-KIE, indicating the latter is relatively better
4. PaddleOCR-VL-XFUND took about 4 hours for 200 steps; PaddleOCR-VL-KIE took 8 hours for 1500 steps. The main difference is that the latter used resized images (smaller file size) for fine-tuning

## ✨️ Model Inference

After fine-tuning, the fine-tuned model can be used for inference. The model can:

1. Output complete information in `JSON` format
2. Output corresponding `JSON` format information based on different input fields

Detailed inference steps have been introduced in [Fine-tuning PaddleOCR-VL: A New Approach — Prompt and Information Extraction](https://aistudio.baidu.com/projectdetail/9857242). Here we recommend a self-developed tool:

### 💫 Using PaddleOCR-VL-REC for Information Extraction

You can use [PaddleOCR-VL-REC](https://github.com/megemini/PaddleOCR-VL-REC) for information extraction:

```python
from paddleocr_vl_rec import PaddleOCRVLRec

# Initialize recognizer
recognizer = PaddleOCRVLRec(
    model_dir="path/to/your/model"
)

# Use dict as query (converted to JSON string)
# Returns JSON format (parsed using json_repair)
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query={"NAME":"", "ITEMS":[]},
    return_json=True
)
# result_json is a dict object
print(type(result_json))  # <class 'dict'>
print(result_json)

# Use list as query (converted to {"item1":"", "item2":""} format)
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query=["item1", "item2"],
    return_json=True
)
print(result_json)

recognizer.close()
```

Additionally, this article uniformly uses llama.cpp server to compare PaddleOCR-VL-KIE, GLM-OCR, and HunyuanOCR.

## ✨️ Evaluation Comparison and Analysis

All comparative experiments uniformly deploy GGUF models using llama.cpp server in a V100 environment, calling the interface via the OpenAI API for recognition.

GLM-OCR and HunyuanOCR's GGUF models are directly available; PaddleOCR-VL-KIE's model was exported locally.

> Note: GLM-OCR's mmproj only has Q8 files; the F16 format was re-exported for comparison.

### 💫 Evaluation Comparison

**Overall Accuracy Ranking**

| Model | Accuracy |
|-------|----------|
| **PaddleOCR-VL-KIE** | **75.99%** |
| PaddleOCR-VL-KIE-1.5 | 68.17% |
| HunyuanOCR | 45.06% |
| GLM-OCR | 44.56% |
| GLM-OCR F16 | 44.42% |
| PaddleOCR-VL-XFUND | 33.72% |

**Per-dataset Accuracy Comparison**

| Dataset | PaddleOCR-VL-KIE | PaddleOCR-VL-KIE-1.5 | HunyuanOCR | GLM-OCR | GLM-OCR F16 | PaddleOCR-VL-XFUND |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| CORD-v2 | 75.00% | 75.60% | 0.00% | 18.25% | 19.05% | 3.37% |
| FATURA | **95.87%** | 84.86% | 56.93% | 59.59% | 59.59% | 47.43% |
| SIBR | 72.33% | 70.04% | 42.15% | 53.52% | 52.80% | 40.24% |
| SROIE | **87.82%** | 87.39% | 80.13% | 61.54% | 60.68% | 26.92% |
| nutritional | **63.72%** | 22.29% | 5.93% | 6.33% | 6.33% | 5.41% |
| wildreceipt | **64.81%** | 48.60% | 42.40% | 36.26% | 36.03% | 31.51% |
| xfund | 47.82% | 52.32% | 35.23% | 32.82% | 32.91% | 39.43% |

**xfund Multilingual Accuracy**

| Language | PaddleOCR-VL-KIE | PaddleOCR-VL-KIE-1.5 | HunyuanOCR | GLM-OCR | GLM-OCR F16 | PaddleOCR-VL-XFUND |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| de | 49.76% | 42.40% | 26.54% | 24.59% | 25.18% | 31.84% |
| es | 41.06% | 41.03% | 25.32% | 32.00% | 32.78% | 35.02% |
| fr | 36.67% | **49.72%** | 37.27% | 32.17% | 32.17% | 36.69% |
| it | 62.59% | 62.28% | 53.46% | 42.28% | 42.54% | 58.25% |
| ja | 53.36% | **54.73%** | 22.77% | 31.00% | 30.57% | 28.40% |
| pt | 39.65% | **57.12%** | 39.75% | 26.48% | 26.23% | 43.83% |
| zh | 50.10% | **63.10%** | 48.67% | 45.39% | 44.84% | 47.46% |

**Key Findings**

1. **PaddleOCR-VL-KIE is the overall best** (75.99%), achieving the best results on FATURA, SROIE, nutritional, and wildreceipt datasets
2. **PaddleOCR-VL-KIE-1.5** (68.17%) outperforms PaddleOCR-VL-KIE on certain xfund languages (fr, ja, pt, zh)
3. **The nutritional dataset shows the largest gap**: PaddleOCR-VL-KIE at 63.72%, while all other models are below 23%
4. **GLM-OCR and GLM-OCR F16** perform nearly identically (44.56% vs 44.42%); F16 precision has no significant impact
5. **PaddleOCR-VL-XFUND** performs worst (33.72%), especially on CORD-v2 and SROIE

To test the model's generalization ability, several external data recognition results were tested.

Evaluation sample count: 129 (50 VAT invoices, 29 motor vehicle invoices, 50 ID cards)

**Overall Accuracy Ranking**

| Model | Accuracy |
|-------|----------|
| **HunyuanOCR** | **80.26%** |
| GLM-OCR | 67.59% |
| PaddleOCR-VL-KIE | 66.83% |
| PaddleOCR-VL-XFUND | 62.97% |
| PaddleOCR-VL-KIE-1.5 | 62.33% |

**Per-category Accuracy Comparison**

| Category | PaddleOCR-VL-KIE | PaddleOCR-VL-KIE-1.5 | HunyuanOCR | GLM-OCR | PaddleOCR-VL-XFUND |
|----------|:---:|:---:|:---:|:---:|:---:|
| VAT invoice | 55.15% | 37.82% | **73.03%** | 53.93% | 50.86% |
| Motor vehicle invoice | 68.79% | 75.39% | **86.20%** | 77.56% | 55.23% |
| ID card | 77.37% | **79.27%** | **84.03%** | 75.47% | 79.57% |

**Key Findings**

1. **PaddleOCR-VL-KIE and GLM-OCR have comparable performance**, demonstrating certain generalization ability
2. **HunyuanOCR performs best on this dataset** (80.26%), ranking first in all three categories. (This performance is likely due to HunyuanOCR being trained on these datasets; see the earlier fine-tuning results for reference.)
3. **PaddleOCR-VL-KIE-1.5 performs worst on VAT invoices** (37.82%), far below PaddleOCR-VL-KIE (55.15%)
4. The **motor vehicle invoice** category shows the largest variation across models, from 55.23% (XFUND) to 86.20% (HunyuanOCR)
5. The **ID card** category shows relatively similar performance across models, all within 75%-84%

As mentioned in the data processing section, the images we use undergo resize scaling. Here we also tested inference performance with original-resolution images:

| Category | Model | Uniform Size | Original Resolution |
|----------|-------|:--------:|:----------:|
| **VAT invoice** | HunyuanOCR | **73.03%** | - |
| | PaddleOCR-VL-KIE | 55.15% | 55.10% |
| | GLM-OCR | 53.93% | 52.87% |
| | PaddleOCR-VL-XFUND | 50.86% | - |
| | PaddleOCR-VL-KIE-1.5 | 37.82% | 39.12% |
| **Motor vehicle invoice** | HunyuanOCR | **86.20%** | - |
| | PaddleOCR-VL-KIE-1.5 | 75.39% | 73.09% |
| | GLM-OCR | 77.56% | 65.21% |
| | PaddleOCR-VL-KIE | 68.79% | 68.40% |
| | PaddleOCR-VL-XFUND | 55.23% | - |
| **ID card** | HunyuanOCR | **84.03%** | - |
| | PaddleOCR-VL-XFUND | 79.57% | - |
| | PaddleOCR-VL-KIE-1.5 | 79.27% | 79.27% |
| | PaddleOCR-VL-KIE | 77.37% | 77.37% |
| | GLM-OCR | 75.47% | 75.47% |

**Key Findings**

Image size has a generally small impact on inference results.

If the above evaluation scores still don't provide a clear picture, refer to [CC-OCR](https://modelscope.cn/datasets/Qwen/CC-OCR) for comparison of LLM KIE evaluation scores:

![](https://ai-studio-static-online.cdn.bcebos.com/e95b278ac6c7457b93244a923ac30cd8d1e31fa5516d41f08fb310635266cde6?v=1)

Additionally, the PaddleOCR-VL model itself does not have KIE capability. If you force it to perform information extraction using prompts, the results are no different from plain OCR:

![](https://ai-studio-static-online.cdn.bcebos.com/78afe48f8d5446588796e06687f92d8fb1479fdab7ba4f8a9c5256a51e65a0ac?v=1)

### 💫 Error Analysis

Below are some common errors. For more details, please refer to the logs.

**List Prediction Errors**

![](https://ai-studio-static-online.cdn.bcebos.com/c9f0679666c543fd8cd3a60705753663758e3694931c4401b250816a9cec9f3c?v=1)

**KEY-VALUE Relationship Errors**

![](https://ai-studio-static-online.cdn.bcebos.com/6c95711a48f84216a063a597d52f60a08620263e98ee4d128de07daf4dbe2cba?v=1)

**Missing Special Symbols**

![](https://ai-studio-static-online.cdn.bcebos.com/898307e53b3a4f05b7a41ad20786927309c850225582410aae2dd652d05bacb5?v=1)

**Unable to Predict Empty Fields**

![](https://ai-studio-static-online.cdn.bcebos.com/4db3ee1ab84f4281847840faf096037e673c80226d7542d59fb0f1d2b25c8c24?v=1)

**Missing Spaces**

![](https://ai-studio-static-online.cdn.bcebos.com/83ac1f724e494f738250148ccfe98b43be52137e55444f21aa68de8427b86da9?v=1)

## ✨️ Conclusion

This article describes the construction and usage of the OCR-KIE dataset, and fine-tuning the PaddleOCR-VL-1.5/1.6 model on this dataset.

Results show that the fine-tuned PaddleOCR-VL-KIE model has strong information extraction capability, outperforming GLM-OCR and HunyuanOCR, and demonstrates good generalization.

## ✨️ Appendix

llama.cpp commands used for evaluation:

```shell
# PaddleOCR-VL-KIE
./build/bin/llama-server -m /home/aistudio/work/PaddleOCR-VL-KIE-GGUF/PaddleOCR-VL-KIE-GGUF.gguf \
  --mmproj /home/aistudio/work/PaddleOCR-VL-KIE-GGUF/PaddleOCR-VL-KIE-GGUF-mmproj.gguf \
  --port 8111 --host 0.0.0.0 --ctx-size 131072 -n 4096 --temp 0 --jinja

python evaluate_ocr_openai.py \
    --jsonl /home/aistudio/work/dataset/test.jsonl \
    --base_url http://localhost:8111/v1 \
    --model paddleocr \
    --image_root /home/aistudio/work/dataset \
    --output_dir paddleocr-vl-kie

# GLM-OCR
./build/bin/llama-server -m /home/aistudio/work/GLM-OCR-GGUF/GLM-OCR-f16.gguf \
  --mmproj /home/aistudio/work/GLM-OCR-GGUF/mmproj-GLM-OCR-Q8_0.gguf \
  --port 8111 --host 0.0.0.0 --ctx-size 131072 -n 4096 --temp 0 --jinja

python evaluate_ocr_openai.py \
    --jsonl /home/aistudio/work/dataset/test.jsonl \
    --base_url http://localhost:8111/v1 \
    --model glm \
    --image_root /home/aistudio/work/dataset \
    --output_dir glm-ocr

# HunyuanOCR
./build/bin/llama-server -m /home/aistudio/work/HunyuanOCR-GGUF/HunyuanOCR-bf16.gguf \
  --mmproj /home/aistudio/work/HunyuanOCR-GGUF/mmproj-HunyuanOCR-bf16.gguf \
  --port 8111 --host 0.0.0.0 --ctx-size 131072 -n 4096 --temp 0 --jinja

python evaluate_ocr_openai.py \
    --jsonl /home/aistudio/work/dataset/test.jsonl \
    --base_url http://localhost:8111/v1 \
    --model hunyuan \
    --image_root /home/aistudio/work/dataset \
    --output_dir hunyuanocr
```
