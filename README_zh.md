# PaddleOCR-VL-KIE 微调 KIE 数据集实现信息抽取

> 💥💥💥 [PaddleOCR-VL-XFUND](https://aistudio.baidu.com/projectdetail/10253559) **Pro** **Max** 版本模型与 OCR-KIE 数据集
>
> **Pro** 
>
> - 基础模型 PaddleOCR-VL-1.5 -> PaddleOCR-VL-1.6
>
> - LLM 信息抽取 -> `RLHF` 信息抽取脚本
>
> - 提出一种 『基于字符像素密度的图片缩放算法』
>
> **Max**
>
> - 整合 7 份 4 类数据集
>
> - 泛化测试 3 份数据集
>

---

![](https://ai-studio-static-online.cdn.bcebos.com/2afba10ecddd47a49a6ec93e542ce8e30c7079fb0af7458e9fb91cb8152329c1)


![](https://ai-studio-static-online.cdn.bcebos.com/c20368be09c649b0a7a6f6d5637b38bb54224ea47d334f7e80e91dcc53b63333)

> 说明：
>
> 图中 PaddleOCR-VL-KIE 为本次微调模型，基于 PaddleOCR-VL-1.6，PaddleOCR-VL-KIE-1.5 基于 PaddleOCR-VL-1.5，用于比对两款基座模型。
>
> GLM-OCR 默认使用 Q8 量化的 mmproj 模型，GLM-OCR F16 使用 F16 的 mmproj 模型
>

![](https://ai-studio-static-online.cdn.bcebos.com/3bf2f669b0264bceba4f098f2f0127be38824fe52ac547f2be33488f9618978c)

---

## ✨️ 引言

[PaddleOCR-VL-XFUND](https://aistudio.baidu.com/projectdetail/10253559) 文章中使用 [XFUND: A Multilingual Form Understanding Benchmark](https://github.com/doc-analysis/XFUND) 数据进行的微调任务，实现表单的关键信息抽取。但是，单个数据集微调的模型必然会出现识别倾向，泛化能力不足的问题。

本文将引入 **OCR-KIE** 数据集，此数据集将目前开源领域能够收集到的 **7** 个数据集，分属 **4** 个领域进行和合并整理，基于 **PaddleOCR-VL-1.6** 模型进行微调，得到模型 **PaddleOCR-VL-KIE**，并在 **3** 个未涉及的数据集上进行泛化识别能力的验证。

结果表明：

- PaddleOCR-VL-KIE 以较大的成绩 75.99% 领先第二名 HunyuanOCR 的 45.06%
- PaddleOCR-VL-KIE 的泛化识别能力以 66.83% 的成绩与 GLM-OCR 的 67.59% 基本相当

> 说明：
>
> 在本文会看到，模型在 XFUND 数据集上的识别能力与 [PaddleOCR-VL-XFUND](https://aistudio.baidu.com/projectdetail/10253559) 相差较大，
>
> 这是因为，OCR-KIE 数据集对 XFUND 以及其他数据集做了一些处理，如图片缩放操作等，并且，未按照 XFUND 原有的 train 与 val 数据集进行数据划分，以防止有的模型利用验证集和测试集训练模型从而导致分数虚高的问题，
>
> 因此，本文中的测试成绩只对于 OCR-KIE 数据集有参考意义，不能与原始数据集中的评测结果进行比较。
>

---

## ✨️ 数据准备 OCR-KIE

本文引入 **OCR-KIE** 数据集。

### 💫 数据集信息

原始数据的信息如下：

| Dataset | Category | Description | Records |
|---------|----------|-------------|---------|
| FATURA | Invoices | Invoice documents (50 templates × 50 instances) | 2,500 |
| SROIE | Receipts | Receipt information extraction | 973 |
| wildreceipt | Receipts | Wild receipt recognition | 1,739 |
| CORD-v2 | Receipts | Commercial receipts | 1,000 |
| xfund | Forms | Multi-language form understanding (7 languages) | 1,393 |
| SIBR | Forms | Structured information in documents | 1,000 |
| nutritional-data-poie-1 | Labels | Nutritional information labels | 483 |

根据图片的类别分为：Invoices，Receipts，Forms，Labels 四类。

![](https://ai-studio-static-online.cdn.bcebos.com/03cd12faa9f94b95b6c1d89d2f6f83d933ee2a2d717248f494c3de5cded14c84)

对于各个数据集做了以下处理：

1. 利用原始数据集中的标注，通过 RLHF 方式编写脚本，统一将标注整理为微调所需的 JSON 格式
2. FATURA 原始数据集有 50 个模板，每个模板 200 条记录，为了保证整体数据集的数据均衡，这里只抽取了每个模板 50 条记录，共计 2500 条
3. 将 PNG 格式的图片转换为体积更小的 JPG 格式
4. 利用本文后面提到的 『基于字符像素密度的图片缩放算法』 以及 SWT 算法对图片进行等比例缩小，从而较小显存占用
5. 对 `header` 字段根据不同语言进行翻译后替代 `header`
6. 混合原有的 train，valid，test 数据后重新以 9：1 的方式划分 train 与 test
7. 以抽取部分字段的方式对数据集进行扩展，扩展后 train 训练集共计 48000 条记录
8. 移除掉识别意义不大的 `other` 字段

> 以下为 `header` 翻译规则

| Language | Translation |
|----------|-------------|
| Chinese (zh) | `标题` |
| German (de) | `Kopfzeile` |
| Spanish (es) | `Encabezado` |
| French (fr) | `En-tête` |
| Italian (it) | `Intestazione` |
| Japanese (ja) | `ヘッダー` |
| Portuguese (pt) | `Cabeçalho` |

最终，每条记录的格式为：

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

- `image_info`: 图片的相对路径
- `text_info[0]`: 识别所需的 Prompt
- `text_info[1]`: 需要识别的字段

其中，识别所需的 Prompt 为：

- Original record: `OCR:{}` with all fields in no_mask
- Extended records: `OCR:{"field1": "", "field2": []}` with only selected fields
- Structured mask records: preserves full structure with type-aware placeholders

field 的设计为：

- String value: `"field": ""`
- List of strings: `"field": []`
- List of dicts: `"field": [{"key": ""}]` (保留第一个元素的结构)
- Nested dict: `"field": {"k1": "", "k2": ""}`

作为对比，这里列出 GLM-OCR 与 HunyuanOCR 的 Prompt 设计：

GLM-OCR 为：

- 请按下列JSON格式输出图中信息:{key:"", ...}

HunyuanOCR 为：

- 从图片中提取字段内容: [key1, key2, ...] 并以 JSON 格式返回。

OCR-KIE 的 prompt 的设计相对两者的区别是：

- OCR-KIE 支持全量信息抽取，也就是说，不需要指定 key
- OCR-KIE 支持列表的抽取
- OCR-KIE 的 Prompt 设计的更简短，并且与 PaddleOCR-VL 的 Prompt `OCR：` 有继承关系

### 💫 信息抽取脚本

原始数据集，如，XFUND 标注的数据不是针对 VLM 的格式的，下面是 XFUND 的示例：

```json
{
    "height": 3508, # 图像高度
    "width": 2480,  # 图像宽度
    "ocr_info": [
        {
            "text": "邮政地址:",  # 单个文本内容
            "label": "question", # 文本所属类别a
            "bbox": [261, 802, 483, 859], # 单个文本框
            "id": 54,  # 文本索引
            "linking": [[54, 60]], # 当前文本和其他文本的关系 [question, answer]
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

所以，需要将这 7 个数据集的标注数据整理为微调所需的结构

[PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md) 中有对 PaddleOCR-VL 进行微调任务的数据格式要求：

```json
{
    "image_info": [
        {"matched_text_index": 0, "image_url": "./assets/table_example.jps"},
    ],
    "text_info": [
        {"text": "OCR:", "tag": "mask"},
        {"text": "দডর মথ বধ বকসট একনজর দখই চনত পরল তর অনমন\nঠক পনতই লকয রখছ\nর নচ থকই চচয বলল কশর, “এইই; পযছ! পযছ!'\nওপর", "tag": "no_mask"},
    ]
}
```

在 [PaddleOCR-VL-XFUND](https://aistudio.baidu.com/projectdetail/10253559) 中提出，可以利用 LLM 的方式，让大模型帮助整理各个标注信息，基本的步骤为：

- 通过 XFUND 中的 link 字段构建 KV pair
- 使用大模型将 KV pair 整合为有意义的 json 数据
- 构建 ERNIEKit 的 SFT VL Dataset Format

但是，这样构建的数据集会由于所使用的大模型的不同而出现以下问题：

- 模型幻觉，导致结果错误
- 使用的模型不同，结果出现不一致，无法复现
- 处理速度慢，消耗资源大

后续，尝试过使用多个大模型串联的 Pipeline 方式，评分监督方式等，但是，始终无法解决以上几个问题。

本文采用的是 RLHF 的 `大模型-人类反馈-脚本生成` 的方式进行数据的标注：

1. 使用大模型先根据原始数据集的标注生成一份数据标注的脚本
2. 使用此脚本生成数据标注，然后利用大模型与人类共同审核标注的效果
3. 根据反馈的意见，使用大模型修改脚本，添加标注规则
4. 重复步骤 2 ，直到脚本满足要求

这样生成的标注数据可复现，执行效率高，而且可以通过不断的添加规则完善具体的情况。

![](https://ai-studio-static-online.cdn.bcebos.com/bc2b8aef8c8c4410a998541fddbfb446d82bf352991241219c27f203ab662b28)

### 💫 基于字符像素密度的图片缩放算法

原始数据集中有的图片分辨率很大，比如 XFUND 的数据集，图片普遍在 `2000 * 3000` 以上。

这么大的图片无论是训练还是推理都会占用很大的显存，但是，图片本身实际上不需要这么高的分辨率就可以承载内部的文字信息。

因此，希望能够缩小图片的尺寸而不损失重要的文字信息。

目前还没有很好的此类方法，有类似的算法 SWT（Stroke Width Transform） 可以通过计算文字的线宽实现识别文字的效果。

但是，此算法针对的是普通生活场景，而不是文本类图片，实际测试中也发现，此算法不能很好做到缩小图片的尺寸。

本文提出一种针对文件、文本类的基于字符像素密度的图片缩放算法：

1. 利用 Otsu 阈值将图片的灰色图像进行二值化处理，从而分离出文字部分与背景（多为白色背景）
2. 统计上述二值化图像中的黑色像素数量（文本多为黑色）
3. 将标注中的字符数量除以上述像素数量，得到 `每个字符占用像素数`
4. 由于图像中不只有文字，还会有线条、噪音等，因此，需要将以上字符占用的像素数做一个 margin 折扣，得到 `有效字符占用像素数`
5. 根据设定的 `每个字符占用像素数最小值` 与 `有效字符占用像素数` 计算得到图片缩放比例
6. 设定最大缩放比例、最大图片尺寸等为算法兜底

![](https://ai-studio-static-online.cdn.bcebos.com/91f2278c12c445b19165dd9185531342fe6168562419434cac459aa5322dca5f)


以下是算法的主要代码：

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

以下是在 xFund 数据集（7 种语言，1393 张图像）上的效果

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

## ✨️ 模型微调

微调的过程与可以参考 [PaddleOCR-VL-0.9B SFT](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft_zh.md)

首先安装 ERNIE：

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

然后，修改配置文件并复制覆盖原有配置文件：

```bash
cp work/sft_config/run_ocr_vl_sft_16k.yaml \
  work/ERNIE/examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

下载 PaddleOCR-VL-1.6 模型，这里使用 aistudio：

```bash
aistudio download --model PaddlePaddle/PaddleOCR-VL-1.6 --local_dir PaddleOCR-VL-1.6
```

最后，就是执行微调命令即可，在 AI Studio 的 A100 环境中进行微调。

> V100 环境无法执行微调，但是可以进行模型推理

```bash
cd work/ERNIE; CUDA_VISIBLE_DEVICES=0 python -m erniekit.launcher train examples/configs/PaddleOCR-VL/sft/run_ocr_vl_sft_16k.yaml
```

以下是训练的日志：

![](https://ai-studio-static-online.cdn.bcebos.com/36e4b973664d447a8fd7863055604aeac7e401b8754347a3b9c41ee535abe37b)

1. PaddleOCR-VL-1.5/1.6 微调的 loss 基本一致
2. PaddleOCR-VL-XFUND 的初始 loss 较低，因为它是基于 PaddleOCR-VL-Receipt 模型微调的
3. PaddleOCR-VL-XFUND 的最终 loss 要大于 PaddleOCR-VL-KIE，说明后者相对来说好一些
4. PaddleOCR-VL-XFUND 200 个 step 用时约 4 小时，PaddleOCR-VL-KIE 1500 个 step 用时 8 小时，最大的区别是，后者用的是 resize 之后的图片（体积小）进行的微调

## ✨️ 模型推理

微调完成后，可以使用微调后的模型进行推理。模型可以：

1. 输出 `JSON` 格式的完整信息
2. 根据不同的输入字段，输出对应的 `JSON` 格式的信息

具体的推理步骤在 [微调 PaddleOCR-VL 新姿势 -- Prompt 与 信息抽取](https://aistudio.baidu.com/projectdetail/9857242) 这篇文章中已经有详细的介绍，这里只单独安利一下自己开发的小工具：

### 💫 使用 PaddleOCR-VL-REC 进行信息抽取

可以使用 [PaddleOCR-VL-REC](https://github.com/megemini/PaddleOCR-VL-REC) 进行信息抽取：

```python
from paddleocr_vl_rec import PaddleOCRVLRec

# 初始化识别器
recognizer = PaddleOCRVLRec(
    model_dir="path/to/your/model"
)

# 使用 dict 作为 query（会被转化为 JSON 字符串）
# 返回 JSON 格式（使用 json_repair 解析结果）
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query={"NAME":"", "ITEMS":[]},
    return_json=True
)
# result_json 是一个字典对象
print(type(result_json))  # <class 'dict'>
print(result_json)

# 使用 list 作为 query（会被转化为 {"item1":"", "item2":""} 的形式）
result_json = recognizer.predict(
    image="/path/to/your/image.jpg",
    query=["item1", "item2"],
    return_json=True
)
print(result_json)

recognizer.close()

```

另外，本文后续统一使用 llama.cpp server 对比 PaddleOCR-VL-KIE, GLM-OCR 和 HunyuanOCR。

## ✨️ 评测对比与分析

本次对比实验统一在 V100 环境中使用 llama.cpp server 的方式部署 GGUF 模型，通过 openai api 调用接口进行识别。

GLM-OCR 与 HunyuanOCR 的 GGUF 模型可以直接获取，PaddleOCR-VL-KIE 的模型在本地进行导出。

> 说明：GLM-OCR 的 mmproj 只有 Q8 的文件，这里重新导出 F16 格式做对比

### 💫 评测对比

**总体准确率排名**

| 模型 | 准确率 |
|------|--------|
| **PaddleOCR-VL-KIE** | **75.99%** |
| PaddleOCR-VL-KIE-1.5 | 68.17% |
| HunyuanOCR | 45.06% |
| GLM-OCR | 44.56% |
| GLM-OCR F16 | 44.42% |
| PaddleOCR-VL-XFUND | 33.72% |

**各数据集准确率对比**

| 数据集 | PaddleOCR-VL-KIE | PaddleOCR-VL-KIE-1.5 | HunyuanOCR | GLM-OCR | GLM-OCR F16 | PaddleOCR-VL-XFUND |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| CORD-v2 | 75.00% | 75.60% | 0.00% | 18.25% | 19.05% | 3.37% |
| FATURA | **95.87%** | 84.86% | 56.93% | 59.59% | 59.59% | 47.43% |
| SIBR | 72.33% | 70.04% | 42.15% | 53.52% | 52.80% | 40.24% |
| SROIE | **87.82%** | 87.39% | 80.13% | 61.54% | 60.68% | 26.92% |
| nutritional | **63.72%** | 22.29% | 5.93% | 6.33% | 6.33% | 5.41% |
| wildreceipt | **64.81%** | 48.60% | 42.40% | 36.26% | 36.03% | 31.51% |
| xfund | 47.82% | 52.32% | 35.23% | 32.82% | 32.91% | 39.43% |

**xfund 多语言准确率**

| 语言 | PaddleOCR-VL-KIE | PaddleOCR-VL-KIE-1.5 | HunyuanOCR | GLM-OCR | GLM-OCR F16 | PaddleOCR-VL-XFUND |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| de | 49.76% | 42.40% | 26.54% | 24.59% | 25.18% | 31.84% |
| es | 41.06% | 41.03% | 25.32% | 32.00% | 32.78% | 35.02% |
| fr | 36.67% | **49.72%** | 37.27% | 32.17% | 32.17% | 36.69% |
| it | 62.59% | 62.28% | 53.46% | 42.28% | 42.54% | 58.25% |
| ja | 53.36% | **54.73%** | 22.77% | 31.00% | 30.57% | 28.40% |
| pt | 39.65% | **57.12%** | 39.75% | 26.48% | 26.23% | 43.83% |
| zh | 50.10% | **63.10%** | 48.67% | 45.39% | 44.84% | 47.46% |

**关键发现**

1. **PaddleOCR-VL-KIE 整体最优**（75.99%），在 FATURA、SROIE、nutritional、wildreceipt 四个数据集上均为最佳
2. **PaddleOCR-VL-KIE-1.5**（68.17%）在 xfund 部分语言（fr、ja、pt、zh）上优于 PaddleOCR-VL-KIE
3. **nutritional 数据集差异最大**：PaddleOCR-VL-KIE 63.72%，其他模型均低于 23%
4. **GLM-OCR 与 GLM-OCR F16** 表现几乎一致（44.56% vs 44.42%），F16 精度无显著影响
5. **PaddleOCR-VL-XFUND** 表现最差（33.72%），在 CORD-v2 和 SROIE 上尤其低

为了测试模型的泛化能力，这里测试了几个外部数据的识别结果。

评测样本数：129（增值税发票 50、机动车发票 29、身份证 50）

**总体准确率排名**

| 模型 | 准确率 |
|------|--------|
| **HunyuanOCR** | **80.26%** |
| GLM-OCR | 67.59% |
| PaddleOCR-VL-KIE | 66.83% |
| PaddleOCR-VL-XFUND | 62.97% |
| PaddleOCR-VL-KIE-1.5 | 62.33% |

**各类别准确率对比**

| 类别 | PaddleOCR-VL-KIE | PaddleOCR-VL-KIE-1.5 | HunyuanOCR | GLM-OCR | PaddleOCR-VL-XFUND |
|------|:---:|:---:|:---:|:---:|:---:|
| 增值税发票 | 55.15% | 37.82% | **73.03%** | 53.93% | 50.86% |
| 机动车发票 | 68.79% | 75.39% | **86.20%** | 77.56% | 55.23% |
| 身份证 | 77.37% | **79.27%** | **84.03%** | 75.47% | 79.57% |

**关键发现**

1. **PaddleOCR-VL-KIE 与 GLM-OCR 成绩相当**，具备一定的泛化能力
2. **HunyuanOCR 在此数据集上表现最优**（80.26%），三个类别均为最佳。（这个成绩的表现很有可能是 HunyuanOCR 在这几个数据集上进行过训练导致，可以参考之前微调的成绩）
3. **PaddleOCR-VL-KIE-1.5 在增值税发票上表现最差**（37.82%），远低于 PaddleOCR-VL-KIE（55.15%）
4. **机动车发票**类别各模型差异最大，从 55.23%（XFUND）到 86.20%（HunyuanOCR）
5. **身份证**类别各模型表现相对接近，均在 75%-84% 之间

在数据处理的部分提到，我们使用的图片会进行 resize 缩小，这里也测试了一下原始尺寸的图片推理的成绩：

| 类别 | 模型 | 统一尺寸 | 原始分辨率 |
|------|------|:--------:|:----------:|
| **增值税发票** | HunyuanOCR | **73.03%** | - |
| | PaddleOCR-VL-KIE | 55.15% | 55.10% |
| | GLM-OCR | 53.93% | 52.87% |
| | PaddleOCR-VL-XFUND | 50.86% | - |
| | PaddleOCR-VL-KIE-1.5 | 37.82% | 39.12% |
| **机动车发票** | HunyuanOCR | **86.20%** | - |
| | PaddleOCR-VL-KIE-1.5 | 75.39% | 73.09% |
| | GLM-OCR | 77.56% | 65.21% |
| | PaddleOCR-VL-KIE | 68.79% | 68.40% |
| | PaddleOCR-VL-XFUND | 55.23% | - |
| **身份证** | HunyuanOCR | **84.03%** | - |
| | PaddleOCR-VL-XFUND | 79.57% | - |
| | PaddleOCR-VL-KIE-1.5 | 79.27% | 79.27% |
| | PaddleOCR-VL-KIE | 77.37% | 77.37% |
| | GLM-OCR | 75.47% | 75.47% |

**关键发现**

图片尺寸对推理的结果影响普遍较小。

如果对以上评测的分数仍然没有很好的概念，可以参考 [CC-OCR](https://modelscope.cn/datasets/Qwen/CC-OCR) 中对于大模型的 KIE 评估的分数对比：

![](https://ai-studio-static-online.cdn.bcebos.com/e95b278ac6c7457b93244a923ac30cd8d1e31fa5516d41f08fb310635266cde6)

另外，PaddleOCR-VL 模型本身是不具备 KIE 能力的，如果强行使用 Prompt 进行信息抽取，效果与单纯的 OCR 没有区别：

![](https://ai-studio-static-online.cdn.bcebos.com/78afe48f8d5446588796e06687f92d8fb1479fdab7ba4f8a9c5256a51e65a0ac)

### 💫 错误分析

这里列举几个常见的错误，更多的信息还请查看日志。

**列表预测错误**

![](https://ai-studio-static-online.cdn.bcebos.com/c9f0679666c543fd8cd3a60705753663758e3694931c4401b250816a9cec9f3c)

**KEY-VALUE 关系错误**

![](https://ai-studio-static-online.cdn.bcebos.com/6c95711a48f84216a063a597d52f60a08620263e98ee4d128de07daf4dbe2cba)

**特殊符号遗漏**

![](https://ai-studio-static-online.cdn.bcebos.com/898307e53b3a4f05b7a41ad20786927309c850225582410aae2dd652d05bacb5)

**不能预测空字段**

![](https://ai-studio-static-online.cdn.bcebos.com/4db3ee1ab84f4281847840faf096037e673c80226d7542d59fb0f1d2b25c8c24)

**遗漏空格**

![](https://ai-studio-static-online.cdn.bcebos.com/83ac1f724e494f738250148ccfe98b43be52137e55444f21aa68de8427b86da9)

## ✨️ 总结

本文介绍了 OCR-KIE 数据集的构建过程与使用，基于 PaddleOCR-VL-1.5/1.6 在此数据集上进行模型微调。

结果表明，微调后的模型 PaddleOCR-VL-KIE 具有较好的信息抽取能力，优于 GLM-OCR 与 HunyuanOCR 模型，并且具有较好的泛化识别能力。

## ✨️ 附录

评测使用的 llama.cpp 命令

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
