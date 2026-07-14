#!/usr/bin/env python3
"""
OCR Model Evaluation Script (OpenAI API)

This script evaluates different OCR models on JSONL datasets using
OpenAI-compatible API for model inference.
"""

import json
import os
import sys
import re
import time
import base64
import json_repair
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

from openai import OpenAI


def image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_no_mask_text(data_item: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    text_info = data_item.get('text_info', [])
    for item in text_info:
        if item.get('tag') == 'no_mask':
            return item.get('text', ''), item.get('tag')
    return None


def extract_image_path(data_item: Dict[str, Any]) -> Optional[str]:
    image_info = data_item.get('image_info', [])
    if image_info and len(image_info) > 0:
        return image_info[0].get('image_url')
    return None


def parse_json_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON text: {e}")
        return None


XFUND_LANG_RE = re.compile(r'xfund/dataset/(\w+)\.(train|val)/')

KNOWN_DATASETS = [
    ('FATURA/', 'FATURA', 'en'),
    ('SROIE_2019_with_labels/', 'SROIE', 'en'),
    ('wildreceipt/', 'wildreceipt', 'en'),
    ('cord-v2/', 'CORD-v2', 'ko'),
    ('xfund/', 'xfund', ''),
    ('nutritional-data-poie-1/', 'nutritional', 'en'),
    ('SIBR/', 'SIBR', 'en'),
]


def extract_dataset_info(image_url: str) -> Tuple[str, str, str]:
    """Extract dataset name and language from image_url."""
    url = image_url.replace('\\', '/')

    if 'xfund' in url:
        m = XFUND_LANG_RE.search(url)
        lang = m.group(1) if m else 'unknown'
        return 'xfund', lang

    for pattern, name, lang in KNOWN_DATASETS:
        if pattern in url:
            return name, lang

    return 'unknown', 'unknown'


def construct_paddleocr_prompt(json_data: Dict[str, Any]) -> str:
    def build_prompt_dict(data: Any) -> Any:
        if isinstance(data, dict):
            return {k: build_prompt_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                return [build_prompt_dict(data[0])]
            return []
        else:
            return ""
    prompt_dict = build_prompt_dict(json_data)
    return f"OCR:{json.dumps(prompt_dict, ensure_ascii=False)}"


def construct_hunyuan_prompt(json_data: Dict[str, Any]) -> str:
    keys = list(json_data.keys())
    return f"从图片中提取字段内容: {str(keys)} 并以 JSON 格式返回。"


def construct_glm_prompt(json_data: Dict[str, Any]) -> str:
    def build_prompt_dict(data: Any) -> Any:
        if isinstance(data, dict):
            return {k: build_prompt_dict(v) for k, v in data.items()}
        else:
            return ""
    prompt_dict = build_prompt_dict(json_data)
    return f"请按下列JSON格式输出图中信息:{json.dumps(prompt_dict, ensure_ascii=False)}"


def compare_sorted_values(expected: Any, predicted: Any) -> bool:
    if isinstance(expected, list) and isinstance(predicted, list):
        try:
            sorted_expected = sorted(expected, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
            sorted_predicted = sorted(predicted, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
            return sorted_expected == sorted_predicted
        except (TypeError, ValueError):
            return expected == predicted
    elif isinstance(expected, dict) and isinstance(predicted, dict):
        if set(expected.keys()) != set(predicted.keys()):
            return False
        return all(compare_sorted_values(expected[k], predicted[k]) for k in expected.keys())
    else:
        return str(expected) == str(predicted)


def compare_results(ground_truth: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    results = {
        'total_fields': 0,
        'correct_fields': 0,
        'field_results': {},
        'accuracy': 0.0
    }

    for key, expected_value in ground_truth.items():
        results['total_fields'] += 1
        predicted_value = prediction.get(key)

        if isinstance(expected_value, dict) and isinstance(predicted_value, dict):
            nested_result = compare_results(expected_value, predicted_value)
            is_correct = nested_result['accuracy'] == 1.0
        elif isinstance(expected_value, list) and isinstance(predicted_value, list):
            is_correct = compare_sorted_values(expected_value, predicted_value)
        else:
            is_correct = str(expected_value) == str(predicted_value)

        results['field_results'][key] = {
            'expected': expected_value,
            'predicted': predicted_value,
            'correct': is_correct
        }

        if is_correct:
            results['correct_fields'] += 1

    if results['total_fields'] > 0:
        results['accuracy'] = results['correct_fields'] / results['total_fields']

    return results


class OCREngine(ABC):
    def __init__(self, base_url: str, api_key: str = "EMPTY"):
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = self.client.models.list().data[0].id

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def recognize(self, image_path: str, prompt: str) -> Dict[str, Any]:
        pass

    def _call_api(self, image_base64: str, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.0,
        )

        return response.choices[0].message.content


class PaddleOCRVL(OCREngine):
    def __init__(self, base_url: str, api_key: str = "EMPTY", prompt_type: str = "paddleocr"):
        super().__init__(base_url, api_key)
        self.prompt_type = prompt_type

    def get_model_name(self) -> str:
        return f"PaddleOCR-VL (OpenAI API)"

    def recognize(self, image_path: str, prompt: str) -> Dict[str, Any]:
        image_base64 = image_to_base64(image_path)
        result_text = self._call_api(image_base64, prompt)

        try:
            result = json_repair.loads(result_text)
        except Exception:
            result = {}

        return result if isinstance(result, dict) else {}


class StatisticsCollector:
    def __init__(self):
        self.dataset_stats = defaultdict(lambda: {
            'total': 0, 'processed': 0, 'failed': 0, 'accuracy_sum': 0.0
        })
        self.xfund_lang_stats = defaultdict(lambda: {
            'total': 0, 'processed': 0, 'failed': 0, 'accuracy_sum': 0.0
        })
        self.total_time = 0.0
        self.sample_times = []

    def record_sample(
        self,
        image_url: str,
        processed: bool,
        accuracy: float = 0.0,
        sample_time: float = 0.0,
    ):
        dataset, lang = extract_dataset_info(image_url)

        self.dataset_stats[dataset]['total'] += 1

        if processed:
            self.dataset_stats[dataset]['processed'] += 1
            self.dataset_stats[dataset]['accuracy_sum'] += accuracy
            self.sample_times.append(sample_time)
            self.total_time += sample_time

            if dataset == 'xfund':
                self.xfund_lang_stats[lang]['total'] += 1
                self.xfund_lang_stats[lang]['processed'] += 1
                self.xfund_lang_stats[lang]['accuracy_sum'] += accuracy
        else:
            self.dataset_stats[dataset]['failed'] += 1
            if dataset == 'xfund':
                self.xfund_lang_stats[lang]['total'] += 1
                self.xfund_lang_stats[lang]['failed'] += 1

    def get_dataset_summary(self) -> Dict[str, Any]:
        summary = {}
        for dataset, stats in sorted(self.dataset_stats.items()):
            avg_accuracy = (
                stats['accuracy_sum'] / stats['processed']
                if stats['processed'] > 0 else 0.0
            )
            summary[dataset] = {
                'total': stats['total'],
                'processed': stats['processed'],
                'failed': stats['failed'],
                'accuracy': avg_accuracy
            }
        return summary

    def get_xfund_lang_summary(self) -> Dict[str, Any]:
        summary = {}
        for lang in ['de', 'es', 'fr', 'it', 'ja', 'pt', 'zh']:
            stats = self.xfund_lang_stats.get(lang)
            if stats is None:
                continue
            avg_accuracy = (
                stats['accuracy_sum'] / stats['processed']
                if stats['processed'] > 0 else 0.0
            )
            summary[lang] = {
                'total': stats['total'],
                'processed': stats['processed'],
                'failed': stats['failed'],
                'accuracy': avg_accuracy
            }
        return summary

    def get_time_stats(self) -> Dict[str, Any]:
        n = len(self.sample_times)
        avg_time = self.total_time / n if n > 0 else 0.0
        return {
            'total_time': self.total_time,
            'sample_count': n,
            'avg_time_per_sample': avg_time,
        }


def resolve_image_path(image_url: str, image_root: Optional[str] = None) -> str:
    """Resolve image path: join with image_root if relative, strip leading ./ """
    if image_root:
        url = image_url.lstrip('./')
        return os.path.join(image_root, url)
    return image_url


def evaluate_dataset(
    jsonl_path: str,
    ocr_engine: OCREngine,
    prompt_type: str = 'paddleocr',
    image_root: Optional[str] = None,
    predictions_path: Optional[str] = None,
    log_path: Optional[str] = None
) -> Dict[str, Any]:
    data = load_jsonl(jsonl_path)

    prompt_constructors = {
        'paddleocr': construct_paddleocr_prompt,
        'hunyuan': construct_hunyuan_prompt,
        'glm': construct_glm_prompt
    }

    if prompt_type not in prompt_constructors:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Supported: {list(prompt_constructors.keys())}")

    construct_prompt = prompt_constructors[prompt_type]

    stats = StatisticsCollector()

    evaluation_results = {
        'model_name': ocr_engine.get_model_name(),
        'prompt_type': prompt_type,
        'total_samples': 0,
        'processed_samples': 0,
        'failed_samples': 0,
        'overall_accuracy': 0.0,
        'sample_results': []
    }

    predictions = []
    log_entries = []
    total_accuracy = 0.0

    for idx, item in enumerate(data):
        evaluation_results['total_samples'] += 1

        text_tag = extract_no_mask_text(item)
        if not text_tag:
            msg = f"Sample {idx}: No 'no_mask' text found, skipping"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            image_path = extract_image_path(item) or ""
            stats.record_sample(image_path, processed=False)
            continue

        text, tag = text_tag

        image_path = extract_image_path(item)
        if not image_path:
            msg = f"Sample {idx}: No image path found, skipping"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            stats.record_sample("", processed=False)
            continue

        image_path = resolve_image_path(image_path, image_root)

        if not os.path.exists(image_path):
            msg = f"Sample {idx}: Image not found: {image_path}, skipping"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            stats.record_sample(image_path, processed=False)
            continue

        json_data = parse_json_text(text)
        if not json_data:
            msg = f"Sample {idx}: Failed to parse JSON text, skipping"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            stats.record_sample(image_path, processed=False)
            continue

        prompt = construct_prompt(json_data)

        try:
            sample_start = time.time()
            prediction = ocr_engine.recognize(image_path, prompt)
            sample_time = time.time() - sample_start
            comparison = compare_results(json_data, prediction)

            sample_result = {
                'sample_id': idx,
                'image_path': image_path,
                'prompt': prompt,
                'ground_truth': json_data,
                'prediction': prediction,
                'accuracy': comparison['accuracy'],
                'field_results': comparison['field_results']
            }

            evaluation_results['sample_results'].append(sample_result)
            total_accuracy += comparison['accuracy']
            evaluation_results['processed_samples'] += 1

            stats.record_sample(
                image_path,
                processed=True,
                accuracy=comparison['accuracy'],
                sample_time=sample_time,
            )

            predictions.append({
                'sample_id': idx,
                'image_path': image_path,
                'ground_truth': json_data,
                'prediction': prediction,
                'accuracy': comparison['accuracy']
            })

            msg = f"Sample {idx}: Accuracy = {comparison['accuracy']:.2%}"
            print(msg)
            log_entries.append(msg)

        except Exception as e:
            msg = f"Sample {idx}: Error during OCR processing: {e}"
            print(msg)
            log_entries.append(msg)
            evaluation_results['failed_samples'] += 1
            stats.record_sample(image_path, processed=False)
            continue

    if evaluation_results['processed_samples'] > 0:
        evaluation_results['overall_accuracy'] = (
            total_accuracy / evaluation_results['processed_samples']
        )

    evaluation_results['dataset_stats'] = stats.get_dataset_summary()
    evaluation_results['xfund_lang_stats'] = stats.get_xfund_lang_summary()
    evaluation_results['time_stats'] = stats.get_time_stats()

    if predictions_path:
        save_predictions(predictions, predictions_path)

    if log_path:
        save_log(log_entries, log_path)

    return evaluation_results


def save_results(results: Dict[str, Any], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {output_path}")


def save_predictions(predictions: List[Dict[str, Any]], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f"Predictions saved to: {output_path}")


def save_log(log_entries: List[str], log_path: str):
    with open(log_path, 'w', encoding='utf-8') as f:
        for entry in log_entries:
            f.write(entry + '\n')
    print(f"Log saved to: {log_path}")


def print_summary(results: Dict[str, Any]):
    time_stats = results.get('time_stats', {})

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {results['model_name']}")
    print(f"Prompt Type: {results['prompt_type']}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Processed Samples: {results['processed_samples']}")
    print(f"Failed Samples: {results['failed_samples']}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"Total Time: {time_stats.get('total_time', 0):.2f}s")
    print(f"Avg Time/Sample: {time_stats.get('avg_time_per_sample', 0):.3f}s")

    print("\n" + "-" * 70)
    print("STATS BY DATASET")
    print("-" * 70)
    print(f"{'Dataset':<20} {'Total':>8} {'Processed':>10} {'Failed':>8} {'Accuracy':>10}")
    print("-" * 70)
    for dataset, stats in results.get('dataset_stats', {}).items():
        print(f"{dataset:<20} {stats['total']:>8} {stats['processed']:>10} {stats['failed']:>8} {stats['accuracy']:>9.2%}")

    xfund_stats = results.get('xfund_lang_stats', {})
    if xfund_stats:
        print("\n" + "-" * 70)
        print("STATS BY XFUND LANGUAGE")
        print("-" * 70)
        print(f"{'Language':<15} {'Total':>8} {'Processed':>10} {'Failed':>8} {'Accuracy':>10}")
        print("-" * 70)
        for lang, stats in xfund_stats.items():
            print(f"{lang:<15} {stats['total']:>8} {stats['processed']:>10} {stats['failed']:>8} {stats['accuracy']:>9.2%}")

    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate OCR models using OpenAI API')
    parser.add_argument('--jsonl', type=str, required=True,
                        help='Path to the JSONL file')
    parser.add_argument('--model', type=str, default='paddleocr',
                        choices=['paddleocr', 'hunyuan', 'glm'],
                        help='OCR model to use (default: paddleocr)')
    parser.add_argument('--base_url', type=str, required=True,
                        help='OpenAI API base URL (e.g., http://0.0.0.0:8111/v1)')
    parser.add_argument('--api_key', type=str, default="EMPTY",
                        help='OpenAI API key (default: EMPTY)')
    parser.add_argument('--image_root', type=str, default=None,
                        help='Root directory for relative image paths in JSONL')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for results (default: output)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    output_predictions_path = os.path.join(args.output_dir, 'predictions.jsonl')
    output_log_path = os.path.join(args.output_dir, 'evaluation.log')

    ocr_engine = PaddleOCRVL(
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_type=args.model
    )

    print(f"Starting evaluation with {ocr_engine.get_model_name()}...")
    print(f"Model: {ocr_engine.model}")
    print(f"API URL: {args.base_url}")
    if args.image_root:
        print(f"Image root: {args.image_root}")
    print(f"Output directory: {args.output_dir}")

    results = evaluate_dataset(
        args.jsonl,
        ocr_engine,
        args.model,
        image_root=args.image_root,
        predictions_path=output_predictions_path,
        log_path=output_log_path
    )

    print_summary(results)
    save_results(results, output_results_path)

    print(f"\nAll results saved to: {args.output_dir}")
    print(f"  - Evaluation results: {output_results_path}")
    print(f"  - Predictions: {output_predictions_path}")
    print(f"  - Log: {output_log_path}")


if __name__ == '__main__':
    main()
