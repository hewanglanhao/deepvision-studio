from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

MAX_VISUALIZATION_SIDE = 56
EDGE_KERNEL_3X3 = [
    [-1.0, -1.0, -1.0],
    [-1.0, 8.0, -1.0],
    [-1.0, -1.0, -1.0],
]


def execute_forward_graph(layers: List[Dict[str, Any]], connections: List[Dict[str, int]], input_tensor: Dict[str, Any] | None) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    validation_issues: List[Dict[str, Any]] = []
    layer_results: List[Dict[str, Any]] = []
    layer_shape_map: Dict[str, str] = {}

    if input_tensor is None:
        return {
            "executionOrder": [],
            "layerResults": [],
            "layerShapeMap": layer_shape_map,
            "finalTensor": None,
            "finalTopK": [],
            "validationIssues": validation_issues,
            "shapePath": [],
            "errors": ["Missing input asset."],
            "warnings": [],
            "resolvedLayers": layers,
        }

    graph = build_execution_graph(layers, connections)
    errors.extend(graph["errors"])
    warnings.extend(graph["warnings"])
    if errors:
        return {
            "executionOrder": [],
            "layerResults": [],
            "layerShapeMap": layer_shape_map,
            "finalTensor": None,
            "finalTopK": [],
            "validationIssues": validation_issues,
            "shapePath": [],
            "errors": errors,
            "warnings": warnings,
            "resolvedLayers": layers,
        }

    topo = topological_sort(graph)
    errors.extend(topo["errors"])
    if errors:
        return {
            "executionOrder": topo["order"],
            "layerResults": [],
            "layerShapeMap": layer_shape_map,
            "finalTensor": None,
            "finalTopK": [],
            "validationIssues": validation_issues,
            "shapePath": [],
            "errors": errors,
            "warnings": warnings,
            "resolvedLayers": layers,
        }

    tensor_by_layer: Dict[int, Dict[str, Any]] = {}
    for layer_id in topo["order"]:
        layer = graph["nodesById"].get(layer_id)
        if layer is None:
            continue

        parent_ids = graph["inbound"].get(layer_id, [])
        parent_tensors = [tensor_by_layer[i] for i in parent_ids if i in tensor_by_layer]
        input_shapes = [t["shape"] for t in parent_tensors]

        issues = validate_layer_params(layer, input_shapes)
        validation_issues.extend(issues)
        layer_warnings = [i["message"] for i in issues if i["severity"] == "warning"]
        layer_errors = [i["message"] for i in issues if i["severity"] == "error"]
        warnings.extend([f"{layer['name']}: {m}" for m in layer_warnings])

        if layer_errors:
            errors.extend([f"{layer['name']}: {m}" for m in layer_errors])
            continue

        try:
            op = execute_operator(layer, parent_tensors, input_tensor)
            tensor_by_layer[layer_id] = op["tensor"]

            output_shape = op["tensor"]["shape"]
            output_shape_label = format_shape_label(output_shape)
            input_shape_label = ", ".join([format_shape_label(s) for s in input_shapes]) if input_shapes else "[]"
            stats = compute_tensor_stats(op["tensor"])
            viz = build_layer_visualization(op["tensor"])

            layer_result = {
                "layerId": layer["id"],
                "layerName": layer["name"],
                "layerType": layer["type"],
                "inputShapes": input_shapes,
                "outputShape": output_shape,
                "inputShapeLabel": input_shape_label,
                "outputShapeLabel": output_shape_label,
                "shapeLabel": output_shape_label,
                "transitionNote": op["transitionNote"],
                "paramsSummary": op["paramsSummary"],
                "warnings": layer_warnings,
                "tensor": op["tensor"],
                "visualization": viz,
                "stats": stats,
            }
            layer_results.append(layer_result)
            layer_shape_map[str(layer["id"])] = output_shape_label
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{layer['name']}: {str(exc)}")

    final_layer = layer_results[-1] if layer_results else None
    final_tensor = final_layer["tensor"] if final_layer else None

    return {
        "executionOrder": topo["order"],
        "layerResults": layer_results,
        "layerShapeMap": layer_shape_map,
        "finalTensor": final_tensor,
        "finalTopK": compute_tensor_stats(final_tensor)["topK"] if final_tensor is not None else [],
        "validationIssues": validation_issues,
        "shapePath": [f"{item['layerName']}: {item['outputShapeLabel']}" for item in layer_results],
        "errors": errors,
        "warnings": warnings,
        "resolvedLayers": layers,
    }


def build_execution_graph(layers: List[Dict[str, Any]], connections: List[Dict[str, int]]) -> Dict[str, Any]:
    nodes_by_id: Dict[int, Dict[str, Any]] = {}
    inbound: Dict[int, List[int]] = {}
    outbound: Dict[int, List[int]] = {}
    errors: List[str] = []
    warnings: List[str] = []

    for layer in layers:
        lid = int(layer["id"])
        if lid in nodes_by_id:
            errors.append(f"Duplicate layer id: {lid}.")
            continue
        nodes_by_id[lid] = layer
        inbound[lid] = []
        outbound[lid] = []

    edge_set = set()

    def add_edge(src: int, dst: int, source: str) -> None:
        if src not in nodes_by_id or dst not in nodes_by_id:
            errors.append(f"Invalid edge {src} -> {dst} from {source}.")
            return
        if src == dst:
            errors.append(f"Self loop is not allowed: {src} -> {dst}.")
            return
        key = f"{src}->{dst}"
        if key in edge_set:
            return
        edge_set.add(key)
        inbound[dst].append(src)
        outbound[src].append(dst)

    for layer in layers:
        for input_id in layer.get("inputs", []):
            add_edge(int(input_id), int(layer["id"]), f"layer({layer['id']}).inputs")

    for edge in connections:
        add_edge(int(edge.get("from", -1)), int(edge.get("to", -1)), "connections")

    for layer in layers:
        has_input = len(inbound.get(int(layer["id"]), [])) > 0
        if layer["type"] != "input" and not has_input:
            warnings.append(f"Layer \"{layer['name']}\" has no inbound edge.")

    return {
        "nodesById": nodes_by_id,
        "inbound": inbound,
        "outbound": outbound,
        "errors": errors,
        "warnings": warnings,
    }


def topological_sort(graph: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    indegree = {node_id: len(arr) for node_id, arr in graph["inbound"].items()}
    queue = [node_id for node_id, degree in indegree.items() if degree == 0]

    order: List[int] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for next_id in graph["outbound"].get(current, []):
            next_degree = indegree.get(next_id, 0) - 1
            indegree[next_id] = next_degree
            if next_degree == 0:
                queue.append(next_id)

    if len(order) != len(graph["nodesById"]):
        errors.append("Graph contains a cycle or disconnected invalid dependency chain.")

    return {"order": order, "errors": errors}


def validate_layer_params(layer: Dict[str, Any], input_shapes: List[List[int]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    input_shape = input_shapes[0] if input_shapes else []

    def issue(severity: str, message: str, field: str | None = None) -> None:
        payload = {
            "layerId": layer["id"],
            "layerName": layer["name"],
            "severity": severity,
            "message": message,
        }
        if field is not None:
            payload["field"] = field
        issues.append(payload)

    if layer.get("enabled", True) is False:
        issue("warning", "Layer is disabled.")
        return issues

    if layer["type"] != "input" and len(input_shapes) == 0:
        issue("error", "Layer has no input tensor.")
        return issues

    if layer["type"] in ("conv2d", "pool2d") and len(input_shape) != 3:
        issue("error", "Conv/Pool requires an image-like input shape [H, W, C].", "inputShape")

    if layer["type"] == "conv2d":
        p = layer["params"]
        if p.get("kernelSize", 0) <= 0:
            issue("error", "kernelSize must be > 0.", "kernelSize")
        if p.get("stride", 0) <= 0:
            issue("error", "stride must be > 0.", "stride")
        if p.get("outChannels", 0) <= 0:
            issue("error", "outChannels must be > 0.", "outChannels")
        out_shape = infer_layer_output_shape(layer, input_shapes)
        if len(out_shape) == 0:
            issue("error", "Invalid output shape. Check kernel/stride/padding/dilation.", "padding")

    if layer["type"] == "pool2d":
        p = layer["params"]
        if p.get("kernelSize", 0) <= 0:
            issue("error", "kernelSize must be > 0.", "kernelSize")
        if p.get("stride", 0) <= 0:
            issue("error", "stride must be > 0.", "stride")
        out_shape = infer_layer_output_shape(layer, input_shapes)
        if len(out_shape) == 0:
            issue("error", "Invalid pool output shape. Check kernel/stride/padding.", "padding")

    if layer["type"] in ("dense", "output"):
        p = layer["params"]
        if p.get("units", 0) <= 0:
            issue("error", "units must be > 0.", "units")
        if len(input_shape) == 0:
            issue("error", "Dense/Output requires a non-empty input shape.", "inputShape")

    if layer["type"] == "dropout":
        rate = layer["params"].get("rate", 0)
        if rate < 0 or rate >= 1:
            issue("error", "dropout rate must be in [0, 1).", "rate")

    return issues


def infer_layer_output_shape(layer: Dict[str, Any], input_shapes: List[List[int]]) -> List[int]:
    input_shape = input_shapes[0] if input_shapes else []
    ltype = layer["type"]
    if ltype == "input":
        p = layer["params"]
        return [p["height"], p["width"], p["channels"]]
    if ltype == "conv2d":
        if len(input_shape) != 3:
            return []
        h, w = input_shape[0], input_shape[1]
        p = layer["params"]
        k = max(1, int(p["kernelSize"]))
        s = max(1, int(p["stride"]))
        pad = max(0, int(p["padding"]))
        d = max(1, int(p["dilation"]))
        effective_k = d * (k - 1) + 1
        out_h = math.floor((h + pad * 2 - effective_k) / s) + 1
        out_w = math.floor((w + pad * 2 - effective_k) / s) + 1
        return [out_h, out_w, max(1, int(p["outChannels"]))] if out_h > 0 and out_w > 0 else []
    if ltype == "pool2d":
        if len(input_shape) != 3:
            return []
        h, w, c = input_shape
        p = layer["params"]
        k = max(1, int(p["kernelSize"]))
        s = max(1, int(p["stride"]))
        pad = max(0, int(p["padding"]))
        out_h = math.floor((h + pad * 2 - k) / s) + 1
        out_w = math.floor((w + pad * 2 - k) / s) + 1
        return [out_h, out_w, c] if out_h > 0 and out_w > 0 else []
    if ltype == "flatten":
        return [shape_element_count(input_shape)]
    if ltype in ("dense", "output"):
        return [max(1, int(layer["params"]["units"]))]
    if ltype in ("activation", "dropout"):
        return input_shape
    return [max(1, int(layer["params"].get("units", 1)))]


def execute_operator(layer: Dict[str, Any], inputs: List[Dict[str, Any]], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    ltype = layer["type"]
    if ltype == "input":
        return run_input_operator(layer, input_tensor)
    if ltype == "conv2d":
        return run_conv2d_operator(layer, inputs[0])
    if ltype == "pool2d":
        return run_pool2d_operator(layer, inputs[0])
    if ltype == "flatten":
        return run_flatten_operator(inputs[0])
    if ltype == "dense":
        return run_dense_operator(layer, inputs[0])
    if ltype == "activation":
        return run_activation_operator(layer, inputs[0])
    if ltype == "dropout":
        return run_dropout_operator(layer, inputs[0])
    if ltype == "output":
        return run_output_operator(layer, inputs[0])
    raise ValueError("Unsupported layer type.")


def run_input_operator(layer: Dict[str, Any], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    p = layer["params"]
    pre = p["preprocessing"]
    return {
        "tensor": {
            **input_tensor,
            "shape": list(input_tensor["shape"]),
            "kind": kind_from_shape(input_tensor["shape"]),
        },
        "transitionNote": "Input tensor enters the graph.",
        "paramsSummary": [
            f"network input: {p['width']}x{p['height']}x{p['channels']}",
            f"preprocess: resize={pre['resizeMode']}, color={pre['colorMode']}, normalize={pre['normalize']}",
        ],
    }


def run_conv2d_operator(layer: Dict[str, Any], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    shape = input_tensor["shape"]
    if len(shape) != 3:
        raise ValueError("Conv2D expects [H, W, C] tensor.")
    h, w, c = shape
    p = layer["params"]
    k = max(1, int(p["kernelSize"]))
    stride = max(1, int(p["stride"]))
    pad = max(0, int(p["padding"]))
    dilation = max(1, int(p["dilation"]))
    out_c = max(1, int(p["outChannels"]))
    effective_k = dilation * (k - 1) + 1
    out_h = math.floor((h + pad * 2 - effective_k) / stride) + 1
    out_w = math.floor((w + pad * 2 - effective_k) / stride) + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Conv2D output shape is invalid.")

    output = [0.0] * (out_h * out_w * out_c)
    for oc in range(out_c):
        kernel_3d = resolve_kernel_3d(layer, oc, c, k)
        bias = float((p.get("bias") or [])[oc]) if oc < len(p.get("bias") or []) else 0.0
        for oy in range(out_h):
            for ox in range(out_w):
                acc = bias
                for ic in range(c):
                    for ky in range(k):
                        for kx in range(k):
                            iy = oy * stride + ky * dilation - pad
                            ix = ox * stride + kx * dilation - pad
                            if iy < 0 or iy >= h or ix < 0 or ix >= w:
                                continue
                            input_val = tensor3d_get(input_tensor["values"], w, c, iy, ix, ic)
                            weight = kernel_3d[ic][ky][kx]
                            acc += input_val * weight
                activated = activate_value(acc, p["activation"])
                tensor3d_set(output, out_w, out_c, oy, ox, oc, activated)

    return {
        "tensor": {
            "kind": "tensor3d",
            "shape": [out_h, out_w, out_c],
            "values": output,
            "colorMode": "grayscale" if out_c == 1 else None,
        },
        "transitionNote": f"conv2d: k={k}, stride={stride}, padding={pad}, dilation={dilation}",
        "paramsSummary": [
            f"outChannels={out_c}",
            f"kernelSize={k}",
            f"stride={stride}",
            f"padding={pad}",
            f"activation={p['activation']}",
        ],
    }


def run_pool2d_operator(layer: Dict[str, Any], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    shape = input_tensor["shape"]
    if len(shape) != 3:
        raise ValueError("Pool2D expects [H, W, C] tensor.")
    h, w, c = shape
    p = layer["params"]
    k = max(1, int(p["kernelSize"]))
    stride = max(1, int(p["stride"]))
    pad = max(0, int(p["padding"]))
    out_h = math.floor((h + pad * 2 - k) / stride) + 1
    out_w = math.floor((w + pad * 2 - k) / stride) + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Pool2D output shape is invalid.")

    output = [0.0] * (out_h * out_w * c)
    for oy in range(out_h):
        for ox in range(out_w):
            for ch in range(c):
                max_val = float("-inf")
                total = 0.0
                cnt = 0
                for ky in range(k):
                    for kx in range(k):
                        iy = oy * stride + ky - pad
                        ix = ox * stride + kx - pad
                        val = 0.0 if iy < 0 or iy >= h or ix < 0 or ix >= w else tensor3d_get(input_tensor["values"], w, c, iy, ix, ch)
                        if val > max_val:
                            max_val = val
                        total += val
                        cnt += 1
                pooled = total / max(1, cnt) if p["mode"] == "avg" else max_val
                tensor3d_set(output, out_w, c, oy, ox, ch, pooled)

    return {
        "tensor": {
            "kind": "tensor3d",
            "shape": [out_h, out_w, c],
            "values": output,
            "colorMode": input_tensor.get("colorMode"),
        },
        "transitionNote": f"pool2d({p['mode']}): k={k}, stride={stride}, padding={pad}",
        "paramsSummary": [
            f"mode={p['mode']}",
            f"kernelSize={k}",
            f"stride={stride}",
            f"padding={pad}",
        ],
    }


def run_flatten_operator(input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tensor": {
            "kind": "vector",
            "shape": [len(input_tensor["values"])],
            "values": list(input_tensor["values"]),
        },
        "transitionNote": "flatten: tensor reshaped into a vector.",
        "paramsSummary": ["explicit flatten layer"],
    }


def run_dense_operator(layer: Dict[str, Any], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    input_vector = input_tensor["values"]
    in_dim = len(input_vector)
    p = layer["params"]
    units = max(1, int(p["units"]))
    out = [0.0] * units

    weights = p.get("weights")
    bias = p.get("bias") or []
    for o in range(units):
        acc = float(bias[o]) if o < len(bias) else 0.0
        for i in range(in_dim):
            w = weights[o][i] if weights and o < len(weights) and i < len(weights[o]) else synthetic_weight(layer["id"], o, i)
            acc += input_vector[i] * w
        out[o] = activate_value(acc, p["activation"])

    return {
        "tensor": {
            "kind": "vector",
            "shape": [units],
            "values": out,
        },
        "transitionNote": f"dense: {in_dim} -> {units}",
        "paramsSummary": [
            f"units={units}",
            f"activation={p['activation']}",
            "weights=custom" if weights else "weights=generated",
        ],
    }


def run_activation_operator(layer: Dict[str, Any], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    activation = layer["params"]["activationType"]
    if activation == "softmax" and len(input_tensor["shape"]) == 1:
        next_values = softmax(input_tensor["values"])
    else:
        next_values = [activate_value(v, activation) for v in input_tensor["values"]]

    tensor = dict(input_tensor)
    tensor["values"] = next_values
    return {
        "tensor": tensor,
        "transitionNote": f"activation: {activation}",
        "paramsSummary": [f"activationType={activation}"],
    }


def run_dropout_operator(layer: Dict[str, Any], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    rate = max(0.0, min(0.95, float(layer["params"]["rate"])))
    training = bool(layer["params"].get("training", False))
    if not training:
        tensor = dict(input_tensor)
        tensor["values"] = list(input_tensor["values"])
        return {
            "tensor": tensor,
            "transitionNote": "dropout skipped in inference mode (training=false).",
            "paramsSummary": [f"rate={rate}", "training=false"],
        }

    keep = 1 - rate
    values = [((0.0 if ((idx + 7) % 5 == 0) else val) / max(keep, 1e-6)) for idx, val in enumerate(input_tensor["values"])]
    tensor = dict(input_tensor)
    tensor["values"] = values
    return {
        "tensor": tensor,
        "transitionNote": "dropout applied in training mode.",
        "paramsSummary": [f"rate={rate}", "training=true"],
    }


def run_output_operator(layer: Dict[str, Any], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    dense_layer = {
        **layer,
        "type": "dense",
        "params": {
            "units": layer["params"]["units"],
            "weights": layer["params"].get("weights"),
            "bias": layer["params"].get("bias"),
            "activation": layer["params"]["activation"],
        },
    }
    dense_out = run_dense_operator(dense_layer, input_tensor)

    tensor = dict(dense_out["tensor"])
    tensor["labels"] = layer["params"].get("labels")
    return {
        "tensor": tensor,
        "transitionNote": f"output: {format_shape_label(input_tensor['shape'])} -> {format_shape_label(tensor['shape'])}",
        "paramsSummary": [
            f"units={layer['params']['units']}",
            f"activation={layer['params']['activation']}",
            f"labels={len(layer['params']['labels'])}" if layer["params"].get("labels") else "labels=none",
        ],
    }


def build_layer_visualization(tensor: Dict[str, Any]) -> Dict[str, Any]:
    shape = tensor["shape"]
    if len(shape) == 3:
        sampled = tensor
        h, w, c = sampled["shape"]
        channel_previews = []
        for channel in range(min(c, 4)):
            values = extract_channel(sampled["values"], h, w, c, channel)
            channel_previews.append(
                {
                    "channel": channel,
                    "width": w,
                    "height": h,
                    "values": normalize_values(values),
                }
            )

        if c == 1:
            main = channel_previews[0]["values"] if channel_previews else []
        elif c == 3 and sampled.get("colorMode") in ("rgb", None):
            main = normalize_values(sampled["values"])
        else:
            main = channel_previews[0]["values"] if channel_previews else []

        return {
            "mode": "image",
            "width": w,
            "height": h,
            "channels": c,
            "values": main,
            "channelPreviews": channel_previews,
        }

    if len(shape) >= 1:
        return {
            "mode": "vector",
            "values": list(tensor["values"][:512]),
        }

    return {"mode": "none", "values": []}


def compute_tensor_stats(tensor: Dict[str, Any] | None) -> Dict[str, Any]:
    values = tensor["values"] if tensor else []
    if len(values) == 0:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "nonZeroRatio": 0,
            "topK": [],
        }

    min_v = float("inf")
    max_v = float("-inf")
    total = 0.0
    non_zero = 0
    for value in values:
        if value < min_v:
            min_v = value
        if value > max_v:
            max_v = value
        total += value
        if abs(value) > 1e-8:
            non_zero += 1

    indexed = []
    labels = (tensor or {}).get("labels")
    for idx, value in enumerate(values):
        indexed.append({
            "index": idx,
            "value": value,
            "label": labels[idx] if isinstance(labels, list) and idx < len(labels) else None,
        })
    indexed.sort(key=lambda item: item["value"], reverse=True)

    return {
        "min": min_v,
        "max": max_v,
        "mean": total / len(values),
        "nonZeroRatio": non_zero / len(values),
        "topK": indexed[:8],
    }


def format_shape_label(shape: List[int]) -> str:
    if len(shape) == 0:
        return "[]"
    return "[" + ", ".join([str(v) for v in shape]) + "]"


def shape_element_count(shape: List[int]) -> int:
    if len(shape) == 0:
        return 0
    out = 1
    for v in shape:
        out *= v
    return out


def kind_from_shape(shape: List[int]) -> str:
    if len(shape) == 0:
        return "scalar"
    if len(shape) == 1:
        return "vector"
    if len(shape) == 2:
        return "matrix"
    return "tensor3d"


def tensor3d_get(values: List[float], w: int, c: int, y: int, x: int, ch: int) -> float:
    idx = ((y * w) + x) * c + ch
    return float(values[idx])


def tensor3d_set(values: List[float], w: int, c: int, y: int, x: int, ch: int, value: float) -> None:
    idx = ((y * w) + x) * c + ch
    values[idx] = value


def resolve_kernel_3d(layer: Dict[str, Any], out_channel: int, in_channels: int, kernel_size: int) -> List[List[List[float]]]:
    kernels = layer["params"].get("kernels")
    kernel = None
    if isinstance(kernels, list) and out_channel < len(kernels):
        kernel = kernels[out_channel].get("weights")

    if isinstance(kernel, list) and len(kernel) > 0:
        out = []
        for in_channel in range(in_channels):
            matrix = None
            if in_channel < len(kernel):
                matrix = kernel[in_channel]
            elif len(kernel) > 0:
                matrix = kernel[-1]
            if matrix is None:
                matrix = layer["params"].get("kernelMatrix") or EDGE_KERNEL_3X3
            out.append(fit_kernel_matrix(matrix, kernel_size))
        return out

    single = fit_kernel_matrix(layer["params"].get("kernelMatrix") or EDGE_KERNEL_3X3, kernel_size)
    return [[row[:] for row in single] for _ in range(in_channels)]


def fit_kernel_matrix(matrix: List[List[float]], kernel_size: int) -> List[List[float]]:
    source = matrix if isinstance(matrix, list) and len(matrix) > 0 else EDGE_KERNEL_3X3
    return [[float(source[y][x]) if y < len(source) and x < len(source[y]) else 0.0 for x in range(kernel_size)] for y in range(kernel_size)]


def synthetic_weight(layer_seed: int, out_index: int, in_index: int) -> float:
    return math.sin((layer_seed + 1) * 0.173 + (out_index + 1) * 0.119 + (in_index + 1) * 0.071) * 0.5


def activate_value(value: float, activation: str) -> float:
    if activation in ("none", "softmax"):
        return value
    if activation == "relu":
        return max(0.0, value)
    if activation == "tanh":
        return math.tanh(value)
    if activation == "gelu":
        cdf = 0.5 * (1 + math.tanh(math.sqrt(2 / math.pi) * (value + 0.044715 * math.pow(value, 3))))
        return value * cdf
    return 1 / (1 + math.exp(-value))


def softmax(values: List[float]) -> List[float]:
    if len(values) == 0:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    if total <= 0:
        return [0.0 for _ in values]
    return [v / total for v in exps]


def normalize_values(values: List[float]) -> List[float]:
    if len(values) == 0:
        return []
    min_v = float("inf")
    max_v = float("-inf")
    for v in values:
        if v < min_v:
            min_v = v
        if v > max_v:
            max_v = v
    span = max(1e-6, max_v - min_v)
    return [(v - min_v) / span for v in values]


def downsample_tensor3d(tensor: Dict[str, Any], max_side: int) -> Dict[str, Any]:
    shape = tensor["shape"]
    if len(shape) != 3:
        return tensor
    h, w, c = shape
    if max(h, w) <= max_side:
        return tensor

    scale = max_side / max(h, w)
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))
    out = [0.0] * (out_h * out_w * c)

    for y in range(out_h):
        src_y = min(h - 1, int(math.floor((y / out_h) * h)))
        for x in range(out_w):
            src_x = min(w - 1, int(math.floor((x / out_w) * w)))
            for ch in range(c):
                src_idx = ((src_y * w) + src_x) * c + ch
                dst_idx = ((y * out_w) + x) * c + ch
                out[dst_idx] = tensor["values"][src_idx]

    sampled = dict(tensor)
    sampled["shape"] = [out_h, out_w, c]
    sampled["values"] = out
    return sampled


def extract_channel(values: List[float], h: int, w: int, c: int, channel: int) -> List[float]:
    out = [0.0] * (h * w)
    for y in range(h):
        for x in range(w):
            source = ((y * w) + x) * c + channel
            out[y * w + x] = values[source]
    return out
