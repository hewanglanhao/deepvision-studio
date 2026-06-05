from __future__ import annotations

import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from forward_engine import execute_forward_graph

app = Flask(__name__)
CORS(app)


@app.get('/api/health')
# 返回服务健康状态，供前端或部署环境探活。
def health():
    return jsonify({'ok': True, 'service': 'forward-backend'})


@app.post('/api/forward')
# 接收前向传播请求并转交给计算服务处理。
def forward():
    payload = request.get_json(silent=True) or {}
    layers = payload.get('layers')
    connections = payload.get('connections')
    input_tensor = payload.get('inputTensor')

    if not isinstance(layers, list) or not isinstance(connections, list):
        return jsonify({'error': 'Invalid payload: layers/connections are required.'}), 400

    try:
        result = execute_forward_graph(layers, connections, input_tensor)
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        return jsonify({'error': str(exc)}), 500


if __name__ == '__main__':
    host = os.getenv('FORWARD_HOST', '127.0.0.1')
    port = int(os.getenv('FORWARD_PORT', '5000'))
    debug = os.getenv('FORWARD_DEBUG', 'false').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
