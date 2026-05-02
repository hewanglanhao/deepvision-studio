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
            for path in sorted((root / label).rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
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


class CsvClassificationDataset(Dataset):
    def __init__(self, path: Path) -> None:
        rows: list[list[str]]
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader)
            rows = [row for row in reader if row]
        label_index = detect_label_column(headers)
        if label_index < 0:
            raise ValueError(f"No label/class/target column found in {path}")
        labels = sorted({row[label_index].strip() for row in rows})
        self.classes = labels
        label_to_id = {label: i for i, label in enumerate(labels)}
        features: list[list[float]] = []
        targets: list[int] = []
        for row in rows:
            values: list[float] = []
            for i, cell in enumerate(row):
                if i == label_index:
                    continue
                values.append(float(cell) if cell.strip() else 0.0)
            features.append(values)
            targets.append(label_to_id[row[label_index].strip()])
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.long)
        self.feature_count = self.x.shape[1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class AutoFlatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            return torch.flatten(x, start_dim=1)
        return x


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request_path = Path(args.request).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    train(request)
    return 0


def train(request: dict[str, Any]) -> None:
    job_id = request["jobId"]
    dataset_root = Path(request["datasetRoot"]).resolve()
    control_file = Path(request["controlFile"]).resolve()
    dataset_id = request["datasetId"]
    split = request["split"]
    config = request["config"]
    layers = request.get("layers") or []

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

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model.parameters(), config)
    scheduler = build_scheduler(optimizer, config, int(config.get("totalEpochs") or 20))

    total_epochs = max(1, int(config.get("totalEpochs") or 20))
    start = time.time()
    for epoch in range(1, total_epochs + 1):
        command = wait_if_paused(control_file)
        if command == "stopped":
            emit({"type": "control", "jobId": job_id, "status": "stopped", "message": "Training stopped."})
            return

        loss, accuracy, gradient_norm = train_epoch(model, train_loader, criterion, optimizer, device, control_file)
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
        "samples": collect_prediction_samples(model, dataset, test_set, device, dataset_root, limit=8),
    })
    emit({"type": "control", "jobId": job_id, "status": "completed", "message": "Training completed. Test set evaluated."})


def load_dataset(dataset_root: Path, dataset_id: str, layers: list[dict[str, Any]]) -> Dataset:
    input_shape = infer_input_shape(layers, dataset_id)
    if dataset_id in {"mnist-1000", "cifar10-500", "cifar10-5000"}:
        image_root = dataset_root / "builtin" / dataset_id / "images"
        return ImageClassificationDataset(image_root, input_shape[0], input_shape[1], input_shape[2])
    if dataset_id == "iris":
        return CsvClassificationDataset(dataset_root / "builtin" / "iris" / "iris.csv")
    if dataset_id == "points-2d":
        return CsvClassificationDataset(dataset_root / "builtin" / "points-2d" / "points.csv")
    raise ValueError(f"Dataset {dataset_id} is not available to the Python trainer.")


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


def build_model(layers: list[dict[str, Any]], sample_x: torch.Tensor, class_count: int) -> nn.Module:
    modules: list[nn.Module] = []
    spatial = sample_x.ndim == 3
    saw_trainable = False
    for layer in layers:
        if layer.get("enabled") is False:
            continue
        layer_type = layer.get("type")
        params = layer.get("params") or {}
        if layer_type == "input":
            continue
        if layer_type == "conv2d":
            modules.append(nn.LazyConv2d(
                out_channels=max(1, int(params.get("outChannels") or 8)),
                kernel_size=max(1, int(params.get("kernelSize") or 3)),
                stride=max(1, int(params.get("stride") or 1)),
                padding=max(0, int(params.get("padding") or 0)),
                dilation=max(1, int(params.get("dilation") or 1)),
            ))
            append_activation(modules, params.get("activation"))
            spatial = True
            saw_trainable = True
        elif layer_type == "pool2d":
            kernel_size = max(1, int(params.get("kernelSize") or 2))
            stride = max(1, int(params.get("stride") or kernel_size))
            padding = max(0, int(params.get("padding") or 0))
            if params.get("mode") == "avg":
                modules.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            else:
                modules.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
            spatial = True
        elif layer_type == "flatten":
            modules.append(AutoFlatten())
            spatial = False
        elif layer_type == "dense":
            if spatial:
                modules.append(AutoFlatten())
                spatial = False
            modules.append(nn.LazyLinear(max(1, int(params.get("units") or 64))))
            append_activation(modules, params.get("activation"))
            saw_trainable = True
        elif layer_type == "activation":
            append_activation(modules, params.get("activationType"))
        elif layer_type == "dropout":
            modules.append(nn.Dropout(p=min(0.9, max(0.0, float(params.get("rate") or 0.2)))))
        elif layer_type == "output":
            if spatial:
                modules.append(AutoFlatten())
                spatial = False
            units = int(params.get("units") or class_count or 2)
            modules.append(nn.LazyLinear(max(1, units)))
            # CrossEntropyLoss expects logits, so softmax is intentionally skipped.
            if params.get("activation") not in {None, "none", "softmax"}:
                append_activation(modules, params.get("activation"))
            saw_trainable = True

    if not saw_trainable:
        if spatial:
            modules.append(AutoFlatten())
        modules.append(nn.LazyLinear(max(2, class_count)))
    model = nn.Sequential(*modules)
    with torch.no_grad():
        model(sample_x.unsqueeze(0))
    return model


def append_activation(modules: list[nn.Module], activation: Any) -> None:
    if activation in {None, "none", "softmax"}:
        return
    if activation == "relu":
        modules.append(nn.ReLU())
    elif activation == "tanh":
        modules.append(nn.Tanh())
    elif activation == "gelu":
        modules.append(nn.GELU())
    elif activation == "sigmoid":
        modules.append(nn.Sigmoid())


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


def build_scheduler(optimizer, config: dict[str, Any], total_epochs: int):
    scheduler = str(config.get("scheduler") or "none").lower()
    decay = float(config.get("lrDecay") or 0.9)
    if scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=decay)
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    return None


def train_epoch(model, loader, criterion, optimizer, device, control_file: Path) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    last_gradient_norm = 0.0
    for x, y in loader:
        command = wait_if_paused(control_file)
        if command == "stopped":
            break
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        last_gradient_norm = compute_gradient_norm(model)
        optimizer.step()
        batch_size = y.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        total += batch_size
    return total_loss / max(1, total), total_correct / max(1, total), last_gradient_norm


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            batch_size = y.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
            total += batch_size
    return total_loss / max(1, total), total_correct / max(1, total)


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
    samples: list[dict[str, Any]] = []
    model.eval()
    indices = list(getattr(subset, "indices", []))[:limit]
    with torch.no_grad():
        for raw_index in indices:
            x, y = dataset[int(raw_index)]
            logits = model(x.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
            pred = int(torch.argmax(probs).item())
            true_index = int(y.item())
            item: dict[str, Any] = {
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


def sample_image_path(dataset: Dataset, index: int) -> Path | None:
    raw_samples = getattr(dataset, "samples", None)
    if not isinstance(raw_samples, list) or index < 0 or index >= len(raw_samples):
        return None
    path = raw_samples[index][0]
    return path if isinstance(path, Path) else Path(path)


def dataset_url(path: Path, dataset_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return ""
    return "/datasets/" + "/".join(quote(part) for part in rel.parts)


def compute_gradient_norm(model) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm(2).cpu()) ** 2
    return math.sqrt(total)


def weight_stats(model) -> tuple[float, float]:
    values = []
    for name, parameter in model.named_parameters():
        if "weight" in name:
            values.append(parameter.detach().flatten().cpu())
    if not values:
        return 0.0, 0.0
    weights = torch.cat(values)
    return float(weights.mean()), float(weights.std(unbiased=False))


def gradient_status(norm: float) -> str:
    if norm < 0.02:
        return "vanishing"
    if norm > 10:
        return "exploding"
    return "stable"


def wait_if_paused(control_file: Path) -> str:
    while True:
        command = read_control(control_file)
        if command != "paused":
            return command
        time.sleep(0.25)


def read_control(control_file: Path) -> str:
    try:
        data = json.loads(control_file.read_text(encoding="utf-8"))
        return str(data.get("command") or "running")
    except Exception:
        return "running"


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


def default_shape(dataset_id: str) -> tuple[int, int, int]:
    if dataset_id in {"cifar10-500", "cifar10-5000"}:
        return 32, 32, 3
    return 28, 28, 1


def detect_label_column(headers: list[str]) -> int:
    candidates = {"label", "labels", "class", "category", "target", "y", "标签", "类别"}
    lowered = [header.strip().lower() for header in headers]
    for i, name in enumerate(lowered):
        if name in candidates:
            return i
    for i, name in enumerate(lowered):
        if any(candidate in name for candidate in candidates):
            return i
    return -1


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
        raise
