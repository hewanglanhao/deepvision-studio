from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


# 包装 PyTorch Sequential，同时保存前端层与 PyTorch module 的映射关系。
class ModelBundle(nn.Sequential):
    def __init__(self, modules: list[nn.Module], layer_refs: list[dict[str, Any]]) -> None:
        super().__init__(*modules)
        self.layer_refs = layer_refs


# 图片分类数据集：从类别文件夹读取图片并转换成 CxHxW 张量。
class ImageClassificationDataset(Dataset):
    def __init__(self, root: Path, width: int, height: int, channels: int) -> None:
        self.root = root
        self.width = width
        self.height = height
        self.channels = channels
        self.samples: list[tuple[Path, int]] = []
        self.classes = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
        if not self.classes:
            raise ValueError(f"No class folders found under {root}")
        for class_index, label in enumerate(self.classes):
            image_paths = [
                path
                for path in sorted((root / label).rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            real_image_paths = [path for path in image_paths if "_lite_" not in path.name.lower()]
            for path in real_image_paths or image_paths:
                self.samples.append((path, class_index))
        if not self.samples:
            raise ValueError(f"No image files found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        mode = "L" if self.channels == 1 else "RGB"
        with Image.open(path) as image:
            image = image.convert(mode)
            if image.size != (self.width, self.height):
                image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)
            data = torch.tensor(list(image.getdata()), dtype=torch.float32) / 255.0
        if self.channels == 1:
            data = data.view(self.height, self.width).unsqueeze(0)
        else:
            data = data.view(self.height, self.width, 3).permute(2, 0, 1)
        return data, torch.tensor(label, dtype=torch.long)


# CSV 分类数据集：解析标签列，数值特征转 float，类别特征做 one-hot 编码。
class CsvClassificationDataset(Dataset):
    def __init__(self, path: Path, label_column: str, class_count: int | None = None) -> None:
        rows: list[list[str]]
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = [row for row in reader if row]
        label_index = resolve_label_column(headers, label_column)
        if label_index < 0:
            raise ValueError(f"Label column {label_column} not found in {path}")
        normalized_rows: list[list[str]] = []
        for row in rows:
            normalized = (row + [""] * len(headers))[:len(headers)]
            if normalized[label_index].strip():
                normalized_rows.append(normalized)
        if not normalized_rows:
            raise ValueError(f"No labeled rows found in {path}")
        self.raw_headers = headers
        self.raw_rows = normalized_rows
        self.label_column = headers[label_index]
        labels = sorted({row[label_index].strip() for row in normalized_rows})
        if class_count is not None and class_count < len(labels):
            raise ValueError(
                f"Configured class count {class_count} is smaller than {len(labels)} labels found in {path}"
            )
        self.classes = labels
        if class_count is not None and class_count > len(labels):
            self.classes = labels + [f"class {i + 1}" for i in range(len(labels), class_count)]
        label_to_id = {label: i for i, label in enumerate(labels)}
        feature_indices = [
            i for i, header in enumerate(headers)
            if i != label_index and not is_ignored_feature_column(header)
        ]
        feature_specs: list[dict[str, Any]] = []
        for i in feature_indices:
            values = [row[i].strip() for row in normalized_rows if row[i].strip()]
            if not values:
                continue
            if all(is_float(value) for value in values):
                feature_specs.append({"kind": "numeric", "index": i})
            else:
                categories = sorted(set(values))
                feature_specs.append({"kind": "categorical", "index": i, "categories": categories})
        if not feature_specs:
            raise ValueError(f"No usable feature columns found in {path}")
        features: list[list[float]] = []
        targets: list[int] = []
        feature_names: list[str] = []
        for spec in feature_specs:
            header = headers[int(spec["index"])]
            if spec["kind"] == "numeric":
                feature_names.append(header)
            else:
                feature_names.extend(f"{header}={category}" for category in list(spec["categories"]))
        for row in normalized_rows:
            values: list[float] = []
            for spec in feature_specs:
                cell = row[int(spec["index"])].strip()
                if spec["kind"] == "numeric":
                    values.append(float(cell) if cell else 0.0)
                else:
                    categories = list(spec["categories"])
                    values.extend(1.0 if cell == category else 0.0 for category in categories)
            features.append(values)
            targets.append(label_to_id[row[label_index].strip()])
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.long)
        self.feature_count = self.x.shape[1]
        self.feature_names = feature_names

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


# 合成回归数据集：为房价回归预设任务提供轻量训练样本。
class SyntheticRegressionDataset(Dataset):
    def __init__(self, count: int = 240, seed: int = 20260518) -> None:
        rng = random.Random(seed)
        rows: list[list[str]] = []
        features: list[list[float]] = []
        targets: list[float] = []
        for index in range(count):
            area = rng.uniform(45.0, 145.0)
            rooms = rng.choice([1, 2, 3, 4, 5])
            age = rng.uniform(0.0, 30.0)
            distance = rng.uniform(0.5, 12.0)
            school_score = rng.uniform(55.0, 98.0)
            noise = rng.gauss(0.0, 10.0)
            price = 28.0 + area * 4.2 + rooms * 18.0 - age * 2.1 - distance * 7.5 + school_score * 1.9 + noise
            price = max(60.0, price)
            rows.append([
                f"{area:.2f}",
                str(rooms),
                f"{age:.2f}",
                f"{distance:.2f}",
                f"{school_score:.2f}",
                f"{price:.2f}",
            ])
            features.append([area / 150.0, rooms / 5.0, age / 30.0, distance / 12.0, school_score / 100.0])
            targets.append(price / 700.0)
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.raw_headers = ["area", "rooms", "age", "distance", "school_score", "price"]
        self.raw_rows = rows
        self.feature_names = self.raw_headers[:-1]
        self.feature_count = self.x.shape[1]
        self.classes = ["price"]
        self.is_regression = True
        self.target_scale = 1.0
        self.target_display_scale = 700.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


# 自动展平层：把图像特征图转成全连接层可接收的向量。
class AutoFlatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            return torch.flatten(x, start_dim=1)
        return x


# 小型残差块：支持 shortcut 投影，保证主分支和旁路维度可相加。
class ResidualBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        activation: Any,
        use_projection: bool,
    ) -> None:
        super().__init__()
        self.conv1 = nn.LazyConv2d(out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = activation_module(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.projection = nn.LazyConv2d(out_channels, kernel_size=1, stride=stride) if use_projection else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x if self.projection is None else self.projection(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        if out.shape != shortcut.shape:
            raise ValueError(
                f"Residual add shape mismatch: main={tuple(out.shape)}, shortcut={tuple(shortcut.shape)}"
            )
        out = out + shortcut
        return self.act(out)


# worker 入口：读取 request.json，根据 action 分发到训练、测试或推理流程。
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("action") == "test_checkpoint":
        test_checkpoint(request)
    elif request.get("action") == "list_checkpoint_samples":
        list_checkpoint_samples(request)
    elif request.get("action") == "infer_checkpoint_sample":
        infer_checkpoint_sample(request)
    else:
        train(request)
    return 0


# 执行一次完整训练任务：加载数据、构建模型、训练验证、保存 checkpoint 并测试。
def train(request: dict[str, Any]) -> None:
    job_id = request["jobId"]
    dataset_root = Path(request["datasetRoot"]).resolve()
    control_file = Path(request["controlFile"]).resolve()
    dataset_id = request["datasetId"]
    split = request["split"]
    config = request["config"]
    layers = request.get("layers") or []
    checkpoint_file = Path(request.get("checkpointFile") or (request_path_fallback(job_id) / "checkpoint.pt")).resolve()
    model_signature = str(request.get("modelSignature") or "")

    seed = int(request.get("seed", 20260427))
    random.seed(seed)
    torch.manual_seed(seed)

    dataset = load_dataset(dataset_root, dataset_id, layers)
    train_set, val_set, test_set = split_dataset(dataset, split, seed)
    batch_size = max(1, int(config.get("batchSize") or 32))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0) if len(val_set) else None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0) if len(test_set) else None
    total_batches = max(1, len(train_loader))

    sample_x, _ = dataset[0]
    class_count = len(getattr(dataset, "classes", []))
    model = build_model(layers, sample_x, class_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = build_criterion(config, dataset)
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, config, int(config.get("totalEpochs") or 20))

    total_epochs = max(1, int(config.get("totalEpochs") or 20))
    start = time.time()
    for epoch in range(1, total_epochs + 1):
        command = wait_if_paused(control_file)
        if command == "stopped":
            emit({"type": "control", "jobId": job_id, "status": "stopped", "message": "Training stopped."})
            return

        loss, accuracy, gradient_norm, backprop = train_epoch(
            model, train_loader, criterion, optimizer, device, control_file, job_id, epoch, total_epochs, total_batches, config
        )
        if scheduler is not None:
            scheduler.step()
        val_loss = None
        val_accuracy = None
        if val_loader is not None:
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        elif test_loader is not None:
            val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)
        weight_mean, weight_std = weight_stats(model)
        elapsed = int(time.time() - start)
        eta = int(elapsed * (total_epochs - epoch) / max(1, epoch))
        lr = optimizer.param_groups[0]["lr"]
        if backprop is not None:
            backprop["phase"] = "validation"
            backprop["lr"] = round(lr, 8)
            backprop["valLoss"] = None if val_loss is None else round(val_loss, 6)
            backprop["valAccuracy"] = None if val_accuracy is None else round(val_accuracy, 6)
            emit(backprop)
        emit({
            "type": "metric",
            "jobId": job_id,
            "step": epoch,
            "epoch": epoch,
            "batch": total_batches,
            "totalEpochs": total_epochs,
            "totalBatches": total_batches,
            "loss": round(loss, 4),
            "valLoss": None if val_loss is None else round(val_loss, 4),
            "accuracy": round(accuracy, 4),
            "valAccuracy": None if val_accuracy is None else round(val_accuracy, 4),
            "lr": round(lr, 8),
            "elapsedSeconds": elapsed,
            "etaSeconds": eta,
            "gradientNorm": round(gradient_norm, 4),
            "weightMean": round(weight_mean, 4),
            "weightStd": round(weight_std, 4),
            "gradientStatus": gradient_status(gradient_norm),
        })

    save_checkpoint(checkpoint_file, model, layers, dataset_id, config, class_count, model_signature)
    test_loss = None
    test_accuracy = None
    if test_loader is not None:
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    emit({
        "type": "test_result",
        "jobId": job_id,
        "testLoss": None if test_loss is None else round(test_loss, 4),
        "testAccuracy": None if test_accuracy is None else round(test_accuracy, 4),
        "sampleCount": len(test_set),
        "checkpointPath": str(checkpoint_file),
        "samples": collect_prediction_samples(model, dataset, test_set, device, dataset_root, limit=8),
    })
    emit({"type": "control", "jobId": job_id, "status": "completed", "message": "Training completed. Test set evaluated."})


# 加载指定 checkpoint 重新跑测试集，用于实验对比页验证历史模型。
def test_checkpoint(request: dict[str, Any]) -> None:
    job_id = request.get("jobId", "checkpoint-test")
    dataset_root = Path(request["datasetRoot"]).resolve()
    dataset_id = request["datasetId"]
    split = request["split"]
    checkpoint_file = Path(request["checkpointFile"]).resolve()

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    layers = checkpoint.get("layers") or request.get("layers") or []
    dataset = load_dataset(dataset_root, dataset_id, layers)
    _, _, test_set = split_dataset(dataset, split, int(request.get("seed", 20260427)))
    if len(test_set) <= 0:
        emit({
            "type": "test_result",
            "jobId": job_id,
            "testLoss": None,
            "testAccuracy": None,
            "sampleCount": 0,
            "samples": [],
        })
        return

    sample_x, _ = dataset[0]
    class_count = len(getattr(dataset, "classes", []))
    model = build_model(layers, sample_x, class_count)
    state = checkpoint.get("modelStateDict") or checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError(f"Checkpoint has no model state: {checkpoint_file}")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = DataLoader(test_set, batch_size=max(1, int(request.get("batchSize") or 32)), shuffle=False, num_workers=0)
    config = checkpoint.get("config") or request.get("config") or {}
    criterion = build_criterion(config, dataset)
    test_loss, test_accuracy = evaluate(model, loader, criterion, device)
    emit({
        "type": "test_result",
        "jobId": job_id,
        "testLoss": round(test_loss, 4),
        "testAccuracy": round(test_accuracy, 4),
        "sampleCount": len(test_set),
        "samples": collect_prediction_samples(model, dataset, test_set, device, dataset_root, limit=8),
    })


# 列出 checkpoint 所属数据集的可推理样本，供前端单样本推理选择。
def list_checkpoint_samples(request: dict[str, Any]) -> None:
    dataset_root = Path(request["datasetRoot"]).resolve()
    dataset_id = request["datasetId"]
    checkpoint_file = Path(request["checkpointFile"]).resolve()
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    layers = checkpoint.get("layers") or request.get("layers") or []
    dataset = load_dataset(dataset_root, dataset_id, layers)
    limit = max(1, min(120, int(request.get("limit") or 60)))
    samples = collect_dataset_samples(dataset, dataset_root, limit)
    emit({
        "type": "sample_list",
        "datasetId": dataset_id,
        "sampleCount": len(dataset),
        "samples": samples,
    })


# 加载 checkpoint 对单个样本做推理，并收集各层激活用于可视化解释。
def infer_checkpoint_sample(request: dict[str, Any]) -> None:
    job_id = request.get("jobId", "single-sample-inference")
    dataset_root = Path(request["datasetRoot"]).resolve()
    dataset_id = request["datasetId"]
    checkpoint_file = Path(request["checkpointFile"]).resolve()
    sample_index = int(request["sampleIndex"])

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    layers = checkpoint.get("layers") or request.get("layers") or []
    dataset = load_dataset(dataset_root, dataset_id, layers)
    if sample_index < 0 or sample_index >= len(dataset):
        raise ValueError(f"Sample index {sample_index} is out of range for dataset {dataset_id}.")

    sample_x, sample_y = dataset[sample_index]
    class_count = len(getattr(dataset, "classes", []))
    model = build_model(layers, sample_x, class_count)
    state = checkpoint.get("modelStateDict") or checkpoint.get("model_state_dict")
    if state is None:
        raise ValueError(f"Checkpoint has no model state: {checkpoint_file}")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits, activations = infer_with_activations(model, sample_x.to(device))
    classes = list(getattr(dataset, "classes", []))
    is_regression = bool(getattr(dataset, "is_regression", False)) or loss_mode(checkpoint.get("config") or {}, dataset) == "mse"
    if is_regression:
        display_scale = float(getattr(dataset, "target_display_scale", 1.0) or 1.0)
        pred_value = float(logits.detach().cpu().view(-1)[0].item())
        true_value = float(sample_y.detach().cpu().view(-1)[0].item())
        predicted_index = 0
        true_index = 0
        confidence = max(0.0, min(1.0, 1.0 - abs(pred_value - true_value) / max(1e-6, float(getattr(dataset, "target_scale", 1.0) or 1.0))))
        true_label = f"{true_value * display_scale:.2f}"
        predicted_label = f"{pred_value * display_scale:.2f}"
        top_k = [{
            "index": 0,
            "label": "predicted value",
            "probability": round(confidence, 6),
        }]
    else:
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        predicted_index = int(torch.argmax(probs).item())
        true_index = int(sample_y.detach().cpu().item())
        confidence = float(probs[predicted_index].item())
        true_label = classes[true_index] if 0 <= true_index < len(classes) else str(true_index)
        predicted_label = classes[predicted_index] if 0 <= predicted_index < len(classes) else str(predicted_index)
        top_k = top_class_predictions(probs, classes, limit=5)
    emit({
        "type": "single_inference",
        "jobId": job_id,
        "datasetId": dataset_id,
        "sample": sample_metadata(dataset, sample_index, dataset_root),
        "prediction": {
            "trueIndex": true_index,
            "trueLabel": true_label,
            "predictedIndex": predicted_index,
            "predictedLabel": predicted_label,
            "confidence": round(float(confidence), 6),
            "correct": abs(float(predicted_label) - float(true_label)) <= 35.0 if is_regression else predicted_index == true_index,
            "topK": top_k,
        },
        "activations": activations,
    })


# 兼容旧请求：当 request 中没有 checkpointFile 时推断默认任务目录。
def request_path_fallback(job_id: str) -> Path:
    return Path.cwd() / "training-jobs" / job_id


# 保存 PyTorch 权重和训练配置，供后续测试、实验对比和单样本推理加载。
def save_checkpoint(
    checkpoint_file: Path,
    model: nn.Module,
    layers: list[dict[str, Any]],
    dataset_id: str,
    config: dict[str, Any],
    class_count: int,
    model_signature: str,
) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "modelStateDict": model.state_dict(),
        "layers": layers,
        "datasetId": dataset_id,
        "config": config,
        "classCount": class_count,
        "modelSignature": model_signature,
    }, checkpoint_file)


# 根据 datasetId 加载内置或上传数据集，并结合输入层推断图片尺寸。
def load_dataset(dataset_root: Path, dataset_id: str, layers: list[dict[str, Any]]) -> Dataset:
    input_shape = infer_input_shape(layers, dataset_id)
    if dataset_id in {"mnist-1000", "cifar10-500", "cifar10-5000"}:
        image_root = dataset_root / "builtin" / dataset_id / "images"
        return ImageClassificationDataset(image_root, input_shape[0], input_shape[1], input_shape[2])
    if dataset_id == "iris":
        return CsvClassificationDataset(dataset_root / "builtin" / "iris" / "iris.csv", "label")
    if dataset_id == "points-2d":
        return CsvClassificationDataset(dataset_root / "builtin" / "points-2d" / "points.csv", "label")
    if dataset_id == "house-price-regression":
        return SyntheticRegressionDataset()
    upload_root = dataset_root / "upload" / dataset_id
    if upload_root.exists():
        image_root = upload_root / "images"
        csv_path = upload_root / "data.csv"
        if image_root.exists():
            return ImageClassificationDataset(image_root, input_shape[0], input_shape[1], input_shape[2])
        if csv_path.exists():
            label_column_path = upload_root / "label-column.txt"
            if not label_column_path.exists():
                raise ValueError(f"Uploaded CSV dataset {dataset_id} has no label-column.txt")
            label_column = label_column_path.read_text(encoding="utf-8").strip()
            class_count_path = upload_root / "class-count.txt"
            class_count = None
            if class_count_path.exists():
                class_count = int(class_count_path.read_text(encoding="utf-8").strip())
            return CsvClassificationDataset(csv_path, label_column, class_count)
    raise ValueError(f"Dataset {dataset_id} is not available to the Python trainer.")


# 按前端配置的 train/val/test 比例和固定 seed 划分数据集。
def split_dataset(dataset: Dataset, split: dict[str, float], seed: int) -> tuple[Subset, Subset, Subset]:
    count = len(dataset)
    train_count = max(1, int(round(count * float(split.get("train", 0.7)))))
    val_count = int(round(count * float(split.get("val", 0.15))))
    if train_count + val_count > count:
        val_count = max(0, count - train_count)
    test_count = max(0, count - train_count - val_count)
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:train_count + val_count + test_count]
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


# 将前端网络层 JSON 动态转换成 PyTorch 模型，并保留 layer_refs 映射。
def build_model(layers: list[dict[str, Any]], sample_x: torch.Tensor, class_count: int) -> nn.Module:
    modules: list[nn.Module] = []
    layer_refs: list[dict[str, Any]] = []
    spatial = sample_x.ndim == 3
    saw_trainable = False
    for layer in layers:
        if layer.get("enabled") is False:
            continue
        layer_type = layer.get("type")
        params = layer.get("params") or {}
        if layer_type == "input":
            layer_refs.append(layer_ref(layer, layer_type, [], trainable=False))
            continue
        if layer_type == "conv2d":
            start = len(modules)
            modules.append(nn.LazyConv2d(
                out_channels=max(1, int(params.get("outChannels") or 8)),
                kernel_size=max(1, int(params.get("kernelSize") or 3)),
                stride=max(1, int(params.get("stride") or 1)),
                padding=max(0, int(params.get("padding") or 0)),
                dilation=max(1, int(params.get("dilation") or 1)),
            ))
            append_activation(modules, params.get("activation"))
            layer_refs.append(layer_ref(layer, layer_type, list(range(start, len(modules))), trainable=True))
            spatial = True
            saw_trainable = True
        elif layer_type == "pool2d":
            start = len(modules)
            kernel_size = max(1, int(params.get("kernelSize") or 2))
            stride = max(1, int(params.get("stride") or kernel_size))
            padding = max(0, int(params.get("padding") or 0))
            if params.get("mode") == "avg":
                modules.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            else:
                modules.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            layer_refs.append(layer_ref(layer, layer_type, list(range(start, len(modules))), trainable=False))
            spatial = True
        elif layer_type == "residual":
            if not spatial:
                raise ValueError(f"Residual block {layer.get('name') or layer.get('id')} requires image-like feature maps.")
            start = len(modules)
            modules.append(ResidualBlock(
                out_channels=max(1, int(params.get("outChannels") or 8)),
                kernel_size=max(1, int(params.get("kernelSize") or 3)),
                stride=max(1, int(params.get("stride") or 1)),
                padding=max(0, int(params.get("padding") or 0)),
                activation=params.get("activation"),
                use_projection=bool(params.get("useProjection")),
            ))
            layer_refs.append(layer_ref(layer, layer_type, list(range(start, len(modules))), trainable=True))
            spatial = True
            saw_trainable = True
        elif layer_type == "flatten":
            start = len(modules)
            modules.append(AutoFlatten())
            layer_refs.append(layer_ref(layer, layer_type, list(range(start, len(modules))), trainable=False))
            spatial = False
        elif layer_type == "dense":
            start = len(modules)
            if spatial:
                modules.append(AutoFlatten())
                spatial = False
            modules.append(nn.LazyLinear(max(1, int(params.get("units") or 64))))
            append_activation(modules, params.get("activation"))
            layer_refs.append(layer_ref(layer, layer_type, list(range(start, len(modules))), trainable=True))
            saw_trainable = True
        elif layer_type == "activation":
            start = len(modules)
            append_activation(modules, params.get("activationType"))
            layer_refs.append(layer_ref(layer, layer_type, list(range(start, len(modules))), trainable=False))
        elif layer_type == "dropout":
            start = len(modules)
            modules.append(nn.Dropout(p=min(0.9, max(0.0, float(params.get("rate") or 0.2)))))
            layer_refs.append(layer_ref(layer, layer_type, list(range(start, len(modules))), trainable=False))
        elif layer_type == "output":
            start = len(modules)
            if spatial:
                modules.append(AutoFlatten())
                spatial = False
            units = int(params.get("units") or class_count or 2)
            modules.append(nn.LazyLinear(max(1, units)))
            # CrossEntropyLoss expects logits, so softmax is intentionally skipped.
            if params.get("activation") not in {None, "none", "softmax"}:
                append_activation(modules, params.get("activation"))
            layer_refs.append(layer_ref(layer, layer_type, list(range(start, len(modules))), trainable=True))
            saw_trainable = True

    if not saw_trainable:
        if spatial:
            modules.append(AutoFlatten())
        modules.append(nn.LazyLinear(max(2, class_count)))
        layer_refs.append({
            "layerId": -1,
            "name": "Auto Output",
            "type": "output",
            "moduleIndices": [len(modules) - 1],
            "trainable": True,
        })
    model = ModelBundle(modules, layer_refs)
    with torch.no_grad():
        model(sample_x.unsqueeze(0))
    return model


# 构造前端 layerId 到 PyTorch module 下标的映射记录。
def layer_ref(layer: dict[str, Any], layer_type: str, module_indices: list[int], trainable: bool) -> dict[str, Any]:
    return {
        "layerId": int(layer.get("id") or -1),
        "name": str(layer.get("name") or layer_type),
        "type": layer_type,
        "moduleIndices": module_indices,
        "trainable": trainable,
    }


# 根据层参数向 module 列表追加激活函数，Identity 不实际加入。
def append_activation(modules: list[nn.Module], activation: Any) -> None:
    module = activation_module(activation)
    if isinstance(module, nn.Identity):
        return
    modules.append(module)


# 将前端激活函数字符串映射为 PyTorch 激活层。
def activation_module(activation: Any) -> nn.Module:
    if activation in {None, "none", "softmax"}:
        return nn.Identity()
    if activation == "relu":
        return nn.ReLU()
    if activation == "tanh":
        return nn.Tanh()
    if activation == "gelu":
        return nn.GELU()
    if activation == "sigmoid":
        return nn.Sigmoid()
    return nn.Identity()


# 根据训练配置创建优化器，支持 SGD、Momentum、Adam、AdamW 等。
def build_optimizer(parameters, config: dict[str, Any]):
    lr = float(config.get("learningRate") or 0.001)
    name = str(config.get("optimizer") or "Adam").lower()
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    if name == "nesterov":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    if name == "adagrad":
        return torch.optim.Adagrad(parameters, lr=lr)
    if name == "adadelta":
        return torch.optim.Adadelta(parameters, lr=lr)
    return torch.optim.Adam(parameters, lr=lr)


# 根据训练配置创建学习率调度器，未启用时返回 None。
def build_scheduler(optimizer, config: dict[str, Any], total_epochs: int):
    scheduler = str(config.get("scheduler") or "none").lower()
    decay = float(config.get("lrDecay") or 0.9)
    if scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=decay)
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    return None


# 从 Dataset 或 DataLoader 包装的 Subset 中安全读取自定义属性。
def dataset_attr(loader_or_dataset: Any, name: str, default: Any = None) -> Any:
    current = loader_or_dataset
    if isinstance(current, DataLoader):
        current = current.dataset
    while hasattr(current, "dataset"):
        current = getattr(current, "dataset")
    return getattr(current, name, default)


# 根据配置和数据集类型判断当前任务是分类、二分类还是回归。
def loss_mode(config: dict[str, Any], dataset: Any) -> str:
    configured = str(config.get("lossFunction") or "").lower()
    if configured in {"mse", "mean_squared_error"} or bool(getattr(dataset, "is_regression", False)):
        return "mse"
    if configured in {"bce", "binary_cross_entropy", "binary_crossentropy"}:
        return "bce"
    return "cross_entropy"


# 根据损失函数实例反推训练目标模式，方便统一计算指标。
def criterion_mode(criterion: nn.Module) -> str:
    if isinstance(criterion, nn.MSELoss):
        return "mse"
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        return "bce"
    return "cross_entropy"


# 根据前端 lossFunction 和数据集类型创建 PyTorch 损失函数。
def build_criterion(config: dict[str, Any], dataset: Dataset) -> nn.Module:
    mode = loss_mode(config, dataset)
    if mode == "mse":
        return nn.MSELoss()
    if mode == "bce":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


# 将标签张量转换成当前损失函数需要的目标格式。
def prepare_targets(logits: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "mse":
        return y.float().view(logits.shape)
    if mode == "bce":
        if logits.ndim == 2 and logits.shape[1] > 1:
            return torch.nn.functional.one_hot(y.long(), num_classes=logits.shape[1]).float()
        return y.float().view_as(logits)
    return y.long()


# 计算一个 batch 的分类准确率或回归近似正确率。
def batch_score(logits: torch.Tensor, y: torch.Tensor, mode: str, target_scale: float = 1.0) -> float:
    if mode == "mse":
        prediction = logits.detach().float().view(-1)
        target = y.detach().float().view(-1)
        mae = torch.abs(prediction - target)
        score = torch.clamp(1.0 - mae / max(1e-6, float(target_scale)), min=0.0, max=1.0)
        return float(score.sum().cpu())
    if mode == "bce" and logits.ndim == 2 and logits.shape[1] == 1:
        pred = (torch.sigmoid(logits).view(-1) >= 0.5).long()
    else:
        pred = logits.argmax(dim=1)
    return float((pred == y.long()).sum().detach().cpu())


# 训练一个 epoch，并在代表 batch 输出前向、损失、反向和参数更新事件。
def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    control_file: Path,
    job_id: str,
    epoch: int,
    total_epochs: int,
    total_batches: int,
    config: dict[str, Any],
) -> tuple[float, float, float, dict[str, Any] | None]:
    model.train()
    mode = criterion_mode(criterion)
    target_scale = float(dataset_attr(loader, "target_scale", 1.0) or 1.0)
    total_loss = 0.0
    total_score = 0.0
    total = 0
    last_gradient_norm = 0.0
    latest_backprop: dict[str, Any] | None = None
    for batch_index, (x, y) in enumerate(loader, start=1):
        command = wait_if_paused(control_file)
        if command == "stopped":
            break
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        target = prepare_targets(logits, y, mode)
        should_emit_phase = batch_index == 1
        if should_emit_phase:
            emit(build_backprop_event(
                job_id,
                epoch,
                total_epochs,
                batch_index,
                total_batches,
                config,
                "forward",
                0.0,
                optimizer.param_groups[0]["lr"],
                logits,
                y,
                [],
            ))
        loss = criterion(logits, target)
        if should_emit_phase:
            emit(build_backprop_event(
                job_id,
                epoch,
                total_epochs,
                batch_index,
                total_batches,
                config,
                "loss",
                float(loss.detach().cpu()),
                optimizer.param_groups[0]["lr"],
                logits,
                y,
                [],
            ))
        before = snapshot_trainable_parameters(model)
        loss.backward()
        last_gradient_norm = compute_gradient_norm(model)
        layer_stats_before_step = collect_layer_backprop_stats(model, before)
        if should_emit_phase:
            emit(build_backprop_event(
                job_id,
                epoch,
                total_epochs,
                batch_index,
                total_batches,
                config,
                "backward",
                float(loss.detach().cpu()),
                optimizer.param_groups[0]["lr"],
                logits,
                y,
                layer_stats_before_step,
            ))
            emit(build_backprop_event(
                job_id,
                epoch,
                total_epochs,
                batch_index,
                total_batches,
                config,
                "gradient_check",
                float(loss.detach().cpu()),
                optimizer.param_groups[0]["lr"],
                logits,
                y,
                layer_stats_before_step,
            ))
        optimizer.step()
        apply_update_norms(layer_stats_before_step, model, before)
        latest_backprop = build_backprop_event(
            job_id,
            epoch,
            total_epochs,
            batch_index,
            total_batches,
            config,
            "optimizer_step",
            float(loss.detach().cpu()),
            optimizer.param_groups[0]["lr"],
            logits,
            y,
            layer_stats_before_step,
        )
        if should_emit_phase:
            emit(latest_backprop)
        batch_size = y.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_score += batch_score(logits, y, mode, target_scale)
        total += batch_size
    return total_loss / max(1, total), total_score / max(1, total), last_gradient_norm, latest_backprop


# 在 optimizer.step 前保存可训练参数快照，用于后续计算参数更新范数。
def snapshot_trainable_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


# 按前端层收集梯度范数、梯度均值、直方图和权重范数等反向传播统计。
def collect_layer_backprop_stats(model: nn.Module, before: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    refs = list(getattr(model, "layer_refs", []))
    rows: list[dict[str, Any]] = []
    modules = list(model)
    for ref in refs:
        params: list[tuple[str, torch.nn.Parameter]] = []
        for module_index in ref.get("moduleIndices", []):
            if not isinstance(module_index, int) or module_index < 0 or module_index >= len(modules):
                continue
            prefix = f"{module_index}."
            params.extend((prefix + name, parameter) for name, parameter in modules[module_index].named_parameters())
        grad_values = [parameter.grad.detach().flatten() for _, parameter in params if parameter.grad is not None]
        weight_values = [parameter.detach().flatten() for _, parameter in params if "weight" in _ and parameter.numel() > 0]
        rows.append({
            "layerId": ref.get("layerId"),
            "name": ref.get("name"),
            "layerType": ref.get("type"),
            "trainable": bool(ref.get("trainable")),
            "gradNorm": round(tensor_norm(grad_values), 6),
            "gradMean": round(tensor_mean(grad_values), 6),
            "gradMax": round(tensor_abs_max(grad_values), 6),
            "gradHistogram": tensor_histogram(grad_values, bins=14),
            "weightNorm": round(tensor_norm(weight_values), 6),
            "updateNorm": 0.0,
            "status": gradient_status(tensor_norm(grad_values)) if grad_values else "no_grad",
            "paramCount": int(sum(parameter.numel() for _, parameter in params)),
        })
    return rows


# optimizer.step 后对比参数快照，补充每层参数更新范数。
def apply_update_norms(rows: list[dict[str, Any]], model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    refs = list(getattr(model, "layer_refs", []))
    modules = list(model)
    for row, ref in zip(rows, refs):
        deltas: list[torch.Tensor] = []
        for module_index in ref.get("moduleIndices", []):
            if not isinstance(module_index, int) or module_index < 0 or module_index >= len(modules):
                continue
            prefix = f"{module_index}."
            for name, parameter in modules[module_index].named_parameters():
                key = prefix + name
                if key in before:
                    deltas.append((parameter.detach() - before[key].to(parameter.device)).flatten())
        row["updateNorm"] = round(tensor_norm(deltas), 6)


# 构造发送给前端的反向传播教学事件，包含阶段、梯度、更新和预测解释。
def build_backprop_event(
    job_id: str,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    config: dict[str, Any],
    phase: str,
    loss: float,
    lr: float,
    logits: torch.Tensor,
    y: torch.Tensor,
    layers: list[dict[str, Any]],
) -> dict[str, Any]:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        first_probs = probs[0].detach().cpu()
        true_index = int(y[0].detach().cpu().item())
        predicted_index = int(torch.argmax(first_probs).item())
        confidence = float(first_probs[predicted_index].item())
        true_probability = float(first_probs[true_index].item()) if 0 <= true_index < first_probs.numel() else 0.0
    global_grad_norm = math.sqrt(sum(float(layer.get("gradNorm") or 0.0) ** 2 for layer in layers))
    global_update_norm = math.sqrt(sum(float(layer.get("updateNorm") or 0.0) ** 2 for layer in layers))
    return {
        "type": "backprop",
        "jobId": job_id,
        "epoch": epoch,
        "totalEpochs": total_epochs,
        "batch": batch,
        "totalBatches": total_batches,
        "phase": phase,
        "loss": round(float(loss), 6),
        "optimizer": str(config.get("optimizer") or "Adam"),
        "scheduler": str(config.get("scheduler") or "none"),
        "learningRate": float(config.get("learningRate") or 0.001),
        "lr": round(float(lr), 8),
        "globalGradNorm": round(global_grad_norm, 6),
        "globalUpdateNorm": round(global_update_norm, 6),
        "gradientStatus": gradient_status(global_grad_norm),
        "layers": layers,
        "prediction": {
            "trueIndex": true_index,
            "predictedIndex": predicted_index,
            "confidence": round(confidence, 6),
            "trueProbability": round(true_probability, 6),
            "correct": predicted_index == true_index,
            "explanation": prediction_explanation(predicted_index, true_index, confidence, true_probability),
        },
    }


# 计算一组张量拼接后的 L2 范数。
def tensor_norm(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(value.detach().norm(2).cpu()) ** 2 for value in values))


# 计算一组张量拼接后的均值。
def tensor_mean(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    merged = torch.cat([value.detach().cpu() for value in values if value.numel() > 0])
    return float(merged.mean()) if merged.numel() else 0.0


# 计算一组张量拼接后的绝对值最大值。
def tensor_abs_max(values: list[torch.Tensor]) -> float:
    if not values:
        return 0.0
    merged = torch.cat([value.detach().abs().cpu() for value in values if value.numel() > 0])
    return float(merged.max()) if merged.numel() else 0.0


# 将梯度张量压缩成直方图数据，供前端反向传播面板展示。
def tensor_histogram(values: list[torch.Tensor], bins: int = 14) -> list[dict[str, Any]]:
    if not values:
        return []
    merged = torch.cat([value.detach().cpu().flatten() for value in values if value.numel() > 0])
    if not merged.numel():
        return []
    min_value = float(merged.min())
    max_value = float(merged.max())
    if math.isclose(min_value, max_value):
        pad = max(1e-6, abs(min_value) * 0.1)
        min_value -= pad
        max_value += pad
    counts = torch.histc(merged, bins=bins, min=min_value, max=max_value).to(torch.int64).tolist()
    width = (max_value - min_value) / max(1, bins)
    return [
        {
            "from": round(min_value + index * width, 6),
            "to": round(min_value + (index + 1) * width, 6),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]


# 根据预测类别、真实类别和置信度生成一句中文教学解释。
def prediction_explanation(predicted_index: int, true_index: int, confidence: float, true_probability: float) -> str:
    if predicted_index == true_index:
        return f"当前样本预测正确，真实类别概率约 {true_probability:.1%}，反向传播会继续巩固这一路输出。"
    return (
        f"当前样本预测为 {predicted_index}，真实类别为 {true_index}；"
        f"反向传播会降低错误类别置信度 {confidence:.1%}，并提高真实类别概率 {true_probability:.1%}。"
    )


# 在验证集或测试集上评估平均 loss 和准确率/回归得分。
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    mode = criterion_mode(criterion)
    target_scale = float(dataset_attr(loader, "target_scale", 1.0) or 1.0)
    total_loss = 0.0
    total_score = 0.0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            target = prepare_targets(logits, y, mode)
            loss = criterion(logits, target)
            batch_size = y.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_score += batch_score(logits, y, mode, target_scale)
            total += batch_size
    return total_loss / max(1, total), total_score / max(1, total)


# 收集少量测试集预测样本，用于前端展示模型正确/错误案例。
def collect_prediction_samples(
    model: nn.Module,
    dataset: Dataset,
    subset: Subset,
    device: torch.device,
    dataset_root: Path,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if len(subset) <= 0:
        return []
    classes = list(getattr(dataset, "classes", []))
    is_regression = bool(getattr(dataset, "is_regression", False))
    display_scale = float(getattr(dataset, "target_display_scale", 1.0) or 1.0)
    samples: list[dict[str, Any]] = []
    model.eval()
    indices = list(getattr(subset, "indices", []))[:limit]
    with torch.no_grad():
        for raw_index in indices:
            x, y = dataset[int(raw_index)]
            logits = model(x.unsqueeze(0).to(device))
            if is_regression:
                pred_value = float(logits.detach().cpu().view(-1)[0].item())
                true_value = float(y.detach().cpu().view(-1)[0].item())
                abs_error = abs(pred_value - true_value)
                score = max(0.0, min(1.0, 1.0 - abs_error / max(1e-6, float(getattr(dataset, "target_scale", 1.0) or 1.0))))
                item: dict[str, Any] = {
                    "index": int(raw_index),
                    "trueIndex": 0,
                    "predictedIndex": 0,
                    "trueLabel": f"{true_value * display_scale:.2f}",
                    "predictedLabel": f"{pred_value * display_scale:.2f}",
                    "confidence": round(score, 4),
                    "correct": abs_error <= 0.05,
                    "absoluteError": round(abs_error * display_scale, 4),
                }
            else:
                probs = torch.softmax(logits, dim=1)[0].detach().cpu()
                pred = int(torch.argmax(probs).item())
                true_index = int(y.item())
                item = {
                    "index": int(raw_index),
                    "trueIndex": true_index,
                    "predictedIndex": pred,
                    "trueLabel": classes[true_index] if 0 <= true_index < len(classes) else str(true_index),
                    "predictedLabel": classes[pred] if 0 <= pred < len(classes) else str(pred),
                    "confidence": round(float(probs[pred].item()), 4),
                    "correct": pred == true_index,
                }
            image_path = sample_image_path(dataset, int(raw_index))
            if image_path is not None:
                item["name"] = image_path.name
                item["imageUrl"] = dataset_url(image_path, dataset_root)
            samples.append(item)
    return samples


# 从数据集中收集样本元信息列表，供单样本推理选择器使用。
def collect_dataset_samples(dataset: Dataset, dataset_root: Path, limit: int = 60) -> list[dict[str, Any]]:
    return [sample_metadata(dataset, index, dataset_root) for index in range(min(limit, len(dataset)))]


# 构造单个样本的标签、索引和图片 URL 等元信息。
def sample_metadata(dataset: Dataset, index: int, dataset_root: Path) -> dict[str, Any]:
    x, y = dataset[index]
    classes = list(getattr(dataset, "classes", []))
    is_regression = bool(getattr(dataset, "is_regression", False))
    display_scale = float(getattr(dataset, "target_display_scale", 1.0) or 1.0)
    true_index = 0 if is_regression else int(y.item())
    true_label = f"{float(y.detach().cpu().view(-1)[0].item()) * display_scale:.2f}" if is_regression else (
        classes[true_index] if 0 <= true_index < len(classes) else str(true_index)
    )
    item: dict[str, Any] = {
        "index": int(index),
        "trueIndex": true_index,
        "trueLabel": true_label,
        "shape": list(x.shape),
    }
    image_path = sample_image_path(dataset, index)
    if image_path is not None:
        item["name"] = image_path.name
        item["imageUrl"] = dataset_url(image_path, dataset_root)
    else:
        values = x.detach().cpu().flatten().tolist()
        raw_headers = list(getattr(dataset, "raw_headers", []))
        raw_rows = list(getattr(dataset, "raw_rows", []))
        raw_row = raw_rows[index] if 0 <= index < len(raw_rows) else []
        item["name"] = f"sample {index}"
        item["featurePreview"] = [round(float(value), 6) for value in values[:18]]
        item["featureNames"] = list(getattr(dataset, "feature_names", []))[:18]
        item["featureCount"] = len(values)
        item["rawHeaders"] = raw_headers
        item["rawValues"] = raw_row
        item["rawPreview"] = [
            {"name": raw_headers[i] if i < len(raw_headers) else f"column {i + 1}", "value": raw_row[i] if i < len(raw_row) else ""}
            for i in range(min(8, max(len(raw_headers), len(raw_row))))
        ]
    return item


# 对单样本逐层前向推理，并捕获每个前端层对应的激活输出。
def infer_with_activations(model: nn.Module, sample_x: torch.Tensor) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    modules = list(model)
    refs = list(getattr(model, "layer_refs", []))
    current = sample_x.unsqueeze(0)
    activations: list[dict[str, Any]] = []
    next_module = 0
    for ref in refs:
        module_indices = [idx for idx in ref.get("moduleIndices", []) if isinstance(idx, int)]
        if module_indices:
            target_module = max(module_indices)
            while next_module <= target_module and next_module < len(modules):
                current = modules[next_module](current)
                next_module += 1
        activations.append(activation_payload(ref, current, len(activations)))
    while next_module < len(modules):
        current = modules[next_module](current)
        next_module += 1
    return current, activations


# 将某层激活张量转换为前端可展示的摘要、预览和 Top 激活值。
def activation_payload(ref: dict[str, Any], tensor: torch.Tensor, order: int) -> dict[str, Any]:
    detached = tensor.detach().cpu()
    without_batch = detached[0] if detached.ndim > 0 and detached.shape[0] == 1 else detached
    flat = without_batch.float().flatten()
    non_zero = float((flat.abs() > 1e-8).float().mean().item()) if flat.numel() else 0.0
    return {
        "order": order,
        "layerId": int(ref.get("layerId") or -1),
        "layerName": str(ref.get("name") or ref.get("type") or "layer"),
        "layerType": str(ref.get("type") or "layer"),
        "shape": list(without_batch.shape),
        "stats": {
            "min": round(float(flat.min().item()), 6) if flat.numel() else 0.0,
            "max": round(float(flat.max().item()), 6) if flat.numel() else 0.0,
            "mean": round(float(flat.mean().item()), 6) if flat.numel() else 0.0,
            "nonZeroRatio": round(non_zero, 6),
        },
        "preview": activation_preview(without_batch),
        "topValues": top_activation_values(flat, limit=6),
    }


# 按张量维度生成灰度图、向量或标量形式的激活预览。
def activation_preview(tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.float()
    if data.ndim == 3:
        channels, height, width = int(data.shape[0]), int(data.shape[1]), int(data.shape[2])
        preview_channels = []
        for channel in range(min(6, channels)):
            values, out_width, out_height = downsample_channel(data[channel], max_side=18)
            preview_channels.append({
                "channel": channel,
                "width": out_width,
                "height": out_height,
                "values": normalize_values(values),
            })
        return {
            "mode": "image",
            "width": width,
            "height": height,
            "channels": channels,
            "channelPreviews": preview_channels,
        }
    values = data.flatten().tolist()
    limited = [round(float(value), 6) for value in values[:40]]
    return {
        "mode": "vector",
        "values": limited,
        "valueCount": len(values),
    }


# 将二维特征图降采样到较小尺寸，减少单样本推理响应体积。
def downsample_channel(channel: torch.Tensor, max_side: int = 18) -> tuple[list[float], int, int]:
    height, width = int(channel.shape[0]), int(channel.shape[1])
    out_height = min(max_side, height)
    out_width = min(max_side, width)
    values: list[float] = []
    for y in range(out_height):
        src_y = min(height - 1, int(y * height / max(1, out_height)))
        for x in range(out_width):
            src_x = min(width - 1, int(x * width / max(1, out_width)))
            values.append(float(channel[src_y, src_x].item()))
    return values, out_width, out_height


# 将激活值归一化到 0 到 1，方便前端绘制热力预览。
def normalize_values(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    span = max(max_value - min_value, 1e-8)
    return [round((value - min_value) / span, 6) for value in values]


# 提取激活张量中数值最大的若干位置和数值。
def top_activation_values(flat: torch.Tensor, limit: int = 6) -> list[dict[str, Any]]:
    if flat.numel() <= 0:
        return []
    values, indices = torch.topk(flat.abs(), k=min(limit, flat.numel()))
    return [
        {
            "index": int(indices[i].item()),
            "value": round(float(flat[int(indices[i].item())].item()), 6),
            "absValue": round(float(values[i].item()), 6),
        }
        for i in range(values.numel())
    ]


# 从类别概率中提取 Top-K 预测结果。
def top_class_predictions(probs: torch.Tensor, classes: list[str], limit: int = 5) -> list[dict[str, Any]]:
    values, indices = torch.topk(probs, k=min(limit, probs.numel()))
    return [
        {
            "index": int(indices[i].item()),
            "label": classes[int(indices[i].item())] if 0 <= int(indices[i].item()) < len(classes) else str(int(indices[i].item())),
            "probability": round(float(values[i].item()), 6),
        }
        for i in range(values.numel())
    ]


# 如果样本来自图片数据集，返回原始图片路径。
def sample_image_path(dataset: Dataset, index: int) -> Path | None:
    raw_samples = getattr(dataset, "samples", None)
    if not isinstance(raw_samples, list) or index < 0 or index >= len(raw_samples):
        return None
    path = raw_samples[index][0]
    return path if isinstance(path, Path) else Path(path)


# 将数据集文件路径转换成 Spring 可代理访问的资源 URL。
def dataset_url(path: Path, dataset_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return ""
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "upload":
        dataset_id = parts[1]
        file_path = "/".join(quote(part) for part in parts[2:])
        return f"/api/training/datasets/{quote(dataset_id)}/files/{file_path}"
    return "/datasets/" + "/".join(quote(part) for part in parts)


# 计算整个模型所有梯度的全局 L2 范数。
def compute_gradient_norm(model) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm(2).cpu()) ** 2
    return math.sqrt(total)


# 统计模型所有参数的均值和标准差，用于权重分布可视化。
def weight_stats(model) -> tuple[float, float]:
    values = []
    for name, parameter in model.named_parameters():
        if "weight" in name:
            values.append(parameter.detach().flatten().cpu())
    if not values:
        return 0.0, 0.0
    weights = torch.cat(values)
    return float(weights.mean()), float(weights.std(unbiased=False))


# 根据梯度范数粗略判断梯度稳定、消失或爆炸。
def gradient_status(norm: float) -> str:
    if norm < 0.02:
        return "vanishing"
    if norm > 10:
        return "exploding"
    return "stable"


# 读取 control.json；如果处于 paused 状态就循环等待恢复或停止命令。
def wait_if_paused(control_file: Path) -> str:
    while True:
        command = read_control(control_file)
        if command != "paused":
            return command
        time.sleep(0.25)


# 安全读取 control.json 中的训练控制命令。
def read_control(control_file: Path) -> str:
    try:
        data = json.loads(control_file.read_text(encoding="utf-8"))
        return str(data.get("command") or "running")
    except Exception:
        return "running"


# 根据输入层参数或数据集默认值推断图片输入宽高和通道数。
def infer_input_shape(layers: list[dict[str, Any]], dataset_id: str) -> tuple[int, int, int]:
    for layer in layers:
        if layer.get("type") == "input":
            params = layer.get("params") or {}
            return (
                max(1, int(params.get("width") or default_shape(dataset_id)[0])),
                max(1, int(params.get("height") or default_shape(dataset_id)[1])),
                max(1, int(params.get("channels") or default_shape(dataset_id)[2])),
            )
    return default_shape(dataset_id)


# 返回内置图像数据集的默认输入尺寸。
def default_shape(dataset_id: str) -> tuple[int, int, int]:
    if dataset_id in {"cifar10-500", "cifar10-5000"}:
        return 32, 32, 3
    return 28, 28, 1


# 在 CSV 表头中查找标签列，支持归一化后的宽松匹配。
def resolve_label_column(headers: list[str], label_column: str) -> int:
    requested = label_column.strip()
    for i, header in enumerate(headers):
        if header.strip() == requested:
            return i
    normalized_requested = normalize_column_name(requested)
    for i, header in enumerate(headers):
        if normalize_column_name(header) == normalized_requested:
            return i
    return -1


# 归一化列名，便于中英文和大小写混合情况下匹配标签列。
def normalize_column_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


# 判断字符串是否可以解析为浮点数。
def is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


# 过滤 ID、姓名等不适合作为训练特征的 CSV 列。
def is_ignored_feature_column(header: str) -> bool:
    normalized = normalize_column_name(header)
    return normalized in {"id", "studentid", "name", "姓名"} or normalized.endswith("id")


# 向 stdout 输出一行 JSON，Spring 后端会逐行读取并转发给前端。
def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        raise
