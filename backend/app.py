from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS

from forward_engine import execute_forward_graph

app = Flask(__name__)
CORS(app)


@app.get('/api/health')
def health():
    return jsonify({'ok': True, 'service': 'forward-backend'})


@app.post('/api/forward')
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
    app.run(host='127.0.0.1', port=5000, debug=True)
