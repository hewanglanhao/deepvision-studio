from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple
import random

import numpy as np

MAX_VISUALIZATION_SIDE = 56
EDGE_KERNEL_3X3 = [
    [-1.0, -1.0, -1.0],
    [-1.0, 8.0, -1.0],
    [-1.0, -1.0, -1.0],
]


# 执行完整前向传播图，返回每层张量、shape、统计信息和校验结果。
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


# 根据前端传来的 layers/connections 建立有向计算图；每条边表示上一层特征张量会输入到下一层继续前向传播。
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

    # 登记一条层间依赖，后续拓扑排序会用它决定张量计算顺序。
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


# 对计算图做拓扑排序，保证 input、conv、pool、dense 等层按依赖顺序执行，避免后层先于上游张量计算。
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


# 校验层参数是否符合深度学习计算约束，例如卷积/池化必须接收 [H, W, C] 特征图。
def validate_layer_params(layer: Dict[str, Any], input_shapes: List[List[int]]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    input_shape = input_shapes[0] if input_shapes else []

    # 把具体参数问题绑定到 layerId/field，前端才能高亮对应层和输入框。
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

    if layer["type"] in ("conv2d", "pool2d", "residual") and len(input_shape) != 3:
        issue("error", "Conv/Pool/Residual requires an image-like input shape [H, W, C].", "inputShape")

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

    if layer["type"] == "residual":
        p = layer["params"]
        if p.get("kernelSize", 0) <= 0:
            issue("error", "kernelSize must be > 0.", "kernelSize")
        if p.get("stride", 0) <= 0:
            issue("error", "stride must be > 0.", "stride")
        if p.get("outChannels", 0) <= 0:
            issue("error", "outChannels must be > 0.", "outChannels")
        out_shape = infer_layer_output_shape(layer, input_shapes)
        if len(out_shape) == 0:
            issue("error", "Invalid residual branch shape. Check kernel/stride/padding.", "padding")
        elif len(input_shape) == 3:
            skip_shape = residual_projection_shape(input_shape, p) if p.get("useProjection") else input_shape
            if skip_shape != out_shape:
                issue(
                    "error",
                    f"Residual add shape mismatch: main={format_shape_label(out_shape)}, shortcut={format_shape_label(skip_shape)}.",
                    "useProjection",
                )

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


# 按层类型推导输出 shape：卷积改变 H/W/C，池化缩小空间尺寸，Flatten 把特征图拉平成向量。
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
    if ltype == "residual":
        if len(input_shape) != 3:
            return []
        h, w, _ = input_shape
        p = layer["params"]
        k = max(1, int(p["kernelSize"]))
        s = max(1, int(p["stride"]))
        pad = max(0, int(p["padding"]))
        mid_h = math.floor((h + pad * 2 - k) / s) + 1
        mid_w = math.floor((w + pad * 2 - k) / s) + 1
        out_h = math.floor((mid_h + pad * 2 - k) / 1) + 1
        out_w = math.floor((mid_w + pad * 2 - k) / 1) + 1
        return [out_h, out_w, max(1, int(p["outChannels"]))] if mid_h > 0 and mid_w > 0 and out_h > 0 and out_w > 0 else []
    if ltype == "flatten":
        return [shape_element_count(input_shape)]
    if ltype in ("dense", "output"):
        return [max(1, int(layer["params"]["units"]))]
    if ltype in ("activation", "dropout"):
        return input_shape
    return [max(1, int(layer["params"].get("units", 1)))]


# 根据 layer.type 分发到具体算子，相当于一个简化版深度学习框架的 operator dispatcher。
def execute_operator(layer: Dict[str, Any], inputs: List[Dict[str, Any]], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    ltype = layer["type"]
    if ltype == "input":
        return run_input_operator(layer, input_tensor)
    if ltype == "conv2d":
        return run_conv2d_operator(layer, inputs[0])
    if ltype == "pool2d":
        return run_pool2d_operator(layer, inputs[0])
    if ltype == "residual":
        return run_residual_operator(layer, inputs[0])
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


# 输入层不做学习计算，只把前端预处理后的图片/表格张量作为整张网络的起点。
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


# 执行 Conv2D：卷积核在局部感受野上滑动，提取边缘、纹理等空间特征，并把输出通道数变成 outChannels。
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

    source = np.asarray(input_tensor["values"], dtype=np.float64).reshape((h, w, c))
    padded = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (effective_k, effective_k),
        axis=(0, 1),
    )[::stride, ::stride, :, ::dilation, ::dilation]
    kernels = np.asarray(
        [resolve_kernel_3d(layer, oc, c, k) for oc in range(out_c)],
        dtype=np.float64,
    ).transpose((1, 2, 3, 0))
    output_arr = np.tensordot(windows, kernels, axes=([2, 3, 4], [0, 1, 2]))
    bias = np.asarray(p.get("bias") or [], dtype=np.float64)
    if bias.size:
        output_arr += np.pad(bias[:out_c], (0, max(0, out_c - bias.size)), mode="constant")
    output_arr = apply_activation_array(output_arr, p["activation"])
    output = output_arr.reshape(-1).tolist()

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


# 执行池化层：在局部窗口内取最大值或平均值，降低特征图分辨率，同时尽量保留主要响应。
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

    source = np.asarray(input_tensor["values"], dtype=np.float64).reshape((h, w, c))
    if p["mode"] == "avg":
        padded = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        mask = np.pad(np.ones((h, w, 1), dtype=np.float64), ((pad, pad), (pad, pad), (0, 0)), mode="constant")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(0, 1))[::stride, ::stride]
        mask_windows = np.lib.stride_tricks.sliding_window_view(mask, (k, k), axis=(0, 1))[::stride, ::stride]
        sums = windows.sum(axis=(-1, -2))
        counts = np.maximum(mask_windows.sum(axis=(-1, -2)), 1.0)
        output_arr = sums / counts
    else:
        padded = np.pad(source, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=-np.inf)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(0, 1))[::stride, ::stride]
        output_arr = windows.max(axis=(-1, -2))
        output_arr = np.where(np.isneginf(output_arr), 0.0, output_arr)
    output = output_arr.reshape(-1).tolist()

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


# 执行残差块：主分支做两次卷积，shortcut 保留原始信息，二者相加后再激活以缓解深层网络退化。
def run_residual_operator(layer: Dict[str, Any], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    shape = input_tensor["shape"]
    if len(shape) != 3:
        raise ValueError("Residual block expects [H, W, C] tensor.")
    p = layer["params"]
    out_c = max(1, int(p["outChannels"]))
    activation = p.get("activation", "relu")

    conv1_layer = {
        **layer,
        "type": "conv2d",
        "name": f"{layer['name']} / conv1",
        "params": {
            "outChannels": out_c,
            "kernelSize": p["kernelSize"],
            "stride": p["stride"],
            "padding": p["padding"],
            "dilation": 1,
            "activation": activation,
        },
    }
    conv1 = run_conv2d_operator(conv1_layer, input_tensor)["tensor"]

    conv2_layer = {
        **layer,
        "type": "conv2d",
        "name": f"{layer['name']} / conv2",
        "params": {
            "outChannels": out_c,
            "kernelSize": p["kernelSize"],
            "stride": 1,
            "padding": p["padding"],
            "dilation": 1,
            "activation": "none",
        },
    }
    main = run_conv2d_operator(conv2_layer, conv1)["tensor"]

    if p.get("useProjection"):
        skip_layer = {
            **layer,
            "type": "conv2d",
            "name": f"{layer['name']} / projection",
            "params": {
                "outChannels": out_c,
                "kernelSize": 1,
                "stride": p["stride"],
                "padding": 0,
                "dilation": 1,
                "activation": "none",
            },
        }
        skip = run_conv2d_operator(skip_layer, input_tensor)["tensor"]
    else:
        skip = {
            **input_tensor,
            "shape": list(input_tensor["shape"]),
            "values": list(input_tensor["values"]),
        }

    if main["shape"] != skip["shape"]:
        raise ValueError(
            f"Residual add shape mismatch: main={format_shape_label(main['shape'])}, shortcut={format_shape_label(skip['shape'])}."
        )

    main_arr = np.asarray(main["values"], dtype=np.float64)
    skip_arr = np.asarray(skip["values"], dtype=np.float64)
    out_arr = apply_activation_array(main_arr + skip_arr, activation)

    return {
        "tensor": {
            "kind": "tensor3d",
            "shape": list(main["shape"]),
            "values": out_arr.tolist(),
            "colorMode": "grayscale" if out_c == 1 else None,
        },
        "transitionNote": f"residual: Conv -> {activation} -> Conv + {'projection' if p.get('useProjection') else 'identity'}",
        "paramsSummary": [
            f"outChannels={out_c}",
            f"kernelSize={max(1, int(p['kernelSize']))}",
            f"stride={max(1, int(p['stride']))}",
            f"padding={max(0, int(p['padding']))}",
            f"projection={bool(p.get('useProjection'))}",
        ],
    }


# 执行 Flatten：把 [H, W, C] 特征图展开成一维向量，连接 CNN 特征提取部分和后面的全连接分类器。
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


# 执行全连接层：用权重矩阵 W 和 bias 综合所有输入特征，形成更抽象的隐藏表示或类别分数。
def run_dense_operator(layer: Dict[str, Any], input_tensor: Dict[str, Any]) -> Dict[str, Any]:
    input_vector = np.asarray(input_tensor["values"], dtype=np.float64)
    in_dim = int(input_vector.size)
    p = layer["params"]
    units = max(1, int(p["units"]))

    weights = p.get("weights")
    weight_matrix = dense_weight_matrix(layer["id"], units, in_dim, weights)
    out_arr = weight_matrix @ input_vector
    bias = np.asarray(p.get("bias") or [], dtype=np.float64)
    if bias.size:
        out_arr += np.pad(bias[:units], (0, max(0, units - bias.size)), mode="constant")

    activation = p["activation"]
    if activation == "softmax":
        out = softmax_array(out_arr).tolist()
    else:
        out = apply_activation_array(out_arr, activation).tolist()

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


# 执行激活层：ReLU/Tanh/GELU 提供非线性表达能力，Softmax 把分类分数转成概率分布。
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


# 执行 Dropout：训练时随机屏蔽一部分神经元以降低过拟合；推理模式下保持张量不变。
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
    rng = random.Random(layer["id"])
    values = [(value / keep) if rng.random() < keep else 0.0 for value in input_tensor["values"]]
    tensor = dict(input_tensor)
    tensor["values"] = values
    return {
        "tensor": tensor,
        "transitionNote": "dropout applied in training mode.",
        "paramsSummary": [f"rate={rate}", "training=true"],
    }


# 执行输出层：本质是最后一个 Dense 层，通常配合 Softmax 输出每个类别的预测概率。
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


# 把层输出张量转换成前端可画的摘要；三维张量显示特征图通道，向量显示神经元响应条形图。
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


# 统计张量最小值、最大值、均值和 Top-K；输出层的 Top-K 可理解为模型最偏向的类别。
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


# 把 shape 转成可读文本，便于对照每层特征图尺寸或向量长度。
def format_shape_label(shape: List[int]) -> str:
    if len(shape) == 0:
        return "[]"
    return "[" + ", ".join([str(v) for v in shape]) + "]"


# 推导残差 shortcut 用 1x1 projection 后的 shape，确保它能和主分支卷积输出相加。
def residual_projection_shape(input_shape: List[int], params: Dict[str, Any]) -> List[int]:
    if len(input_shape) != 3:
        return []
    h, w, _ = input_shape
    stride = max(1, int(params.get("stride", 1)))
    out_channels = max(1, int(params.get("outChannels", 1)))
    out_h = math.floor((h - 1) / stride) + 1
    out_w = math.floor((w - 1) / stride) + 1
    return [out_h, out_w, out_channels] if out_h > 0 and out_w > 0 else []


# 计算张量元素总数，Flatten 输出长度和 Dense 输入维度都依赖这个值。
def shape_element_count(shape: List[int]) -> int:
    if len(shape) == 0:
        return 0
    out = 1
    for v in shape:
        out *= v
    return out


# 根据 shape 维度标记张量类型，区分向量、矩阵和 CNN 常见的三维特征图。
def kind_from_shape(shape: List[int]) -> str:
    if len(shape) == 0:
        return "scalar"
    if len(shape) == 1:
        return "vector"
    if len(shape) == 2:
        return "matrix"
    return "tensor3d"


# 按 [H, W, C] 展平布局读取某个像素位置和通道的特征值。
def tensor3d_get(values: List[float], w: int, c: int, y: int, x: int, ch: int) -> float:
    idx = ((y * w) + x) * c + ch
    return float(values[idx])


# 按 [H, W, C] 展平布局写入某个像素位置和通道的特征值。
def tensor3d_set(values: List[float], w: int, c: int, y: int, x: int, ch: int, value: float) -> None:
    idx = ((y * w) + x) * c + ch
    values[idx] = value


# 解析卷积核权重；多输入通道时，每个输出通道都需要一组跨通道滤波器。
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


# 将预设卷积核适配到当前 kernelSize，支持边缘检测、模糊、锐化等滤波器对比实验。
def fit_kernel_matrix(matrix: List[List[float]], kernel_size: int) -> List[List[float]]:
    source = matrix if isinstance(matrix, list) and len(matrix) > 0 else EDGE_KERNEL_3X3
    return [[float(source[y][x]) if y < len(source) and x < len(source[y]) else 0.0 for x in range(kernel_size)] for y in range(kernel_size)]


# 准备 Dense 层权重矩阵；没有训练权重时生成稳定的演示权重，保证前向结果可重复。
def dense_weight_matrix(layer_seed: int, units: int, in_dim: int, weights: Any) -> np.ndarray:
    out_idx = np.arange(units, dtype=np.float64)[:, None]
    in_idx = np.arange(in_dim, dtype=np.float64)[None, :]
    scale = 1.0 / math.sqrt(max(1, in_dim))
    matrix = np.sin((layer_seed + 1) * 0.173 + (out_idx + 1) * 0.119 + (in_idx + 1) * 0.071) * 0.5 * scale

    if isinstance(weights, list):
        for row_idx, row in enumerate(weights[:units]):
            if not isinstance(row, list):
                continue
            usable = min(in_dim, len(row))
            if usable > 0:
                matrix[row_idx, :usable] = np.asarray(row[:usable], dtype=np.float64)
    return matrix


# 生成单个演示权重，早期逐元素实现保留的辅助函数。
def synthetic_weight(layer_seed: int, out_index: int, in_index: int) -> float:
    return math.sin((layer_seed + 1) * 0.173 + (out_index + 1) * 0.119 + (in_index + 1) * 0.071) * 0.5


# 对整块 NumPy 张量应用激活函数，ReLU 会抑制负响应，Sigmoid/Tanh 会压缩数值范围。
def apply_activation_array(values: np.ndarray, activation: str) -> np.ndarray:
    if activation in ("none", "softmax"):
        return values
    if activation == "relu":
        return np.maximum(values, 0.0)
    if activation == "tanh":
        return np.tanh(values)
    if activation == "gelu":
        cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (values + 0.044715 * np.power(values, 3))))
        return values * cdf
    return 1 / (1 + np.exp(-values))


# 对单个神经元响应应用激活函数，供非向量化路径复用。
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


# 将一组 logits 转成概率分布，常用于分类输出层解释“哪个类别最可能”。
def softmax(values: List[float]) -> List[float]:
    if len(values) == 0:
        return []
    max_v = max(values)
    exps = [math.exp(v - max_v) for v in values]
    total = sum(exps)
    if total <= 0:
        return [0.0 for _ in values]
    return [v / total for v in exps]


# NumPy 版 Softmax；先减最大值提升数值稳定性，避免指数计算溢出。
def softmax_array(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    exps = np.exp(values - np.max(values))
    total = np.sum(exps)
    if total <= 0:
        return np.zeros_like(values)
    return exps / total


# 将任意响应值缩放到 0-1，便于把特征图或热力图映射成可视化颜色。
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


# 对过大的三维特征图做下采样，降低前端展示成本，但不改变真实 forward 计算结果。
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


# 从 [H, W, C] 特征图中抽出单个通道，用来观察某个卷积核的响应模式。
def extract_channel(values: List[float], h: int, w: int, c: int, channel: int) -> List[float]:
    out = [0.0] * (h * w)
    for y in range(h):
        for x in range(w):
            source = ((y * w) + x) * c + channel
            out[y * w + x] = values[source]
    return out
