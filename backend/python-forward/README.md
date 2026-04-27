# Python Forward Backend

这个 Flask 服务只负责 A 模式前向传播计算。前端不直接调用它，而是通过 Spring 的 `POST /api/forward` 代理访问。

## 安装依赖

```powershell
cd backend/python-forward
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 启动

```powershell
python app.py
```

默认地址：`http://127.0.0.1:5000`

- 健康检查：`GET /api/health`
- 前向推理：`POST /api/forward`
