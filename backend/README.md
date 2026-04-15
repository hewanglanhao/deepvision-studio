# Forward Backend (Flask)

仅服务 A 模式前向传播重计算，减少前端主线程卡顿。

## 1) 安装依赖

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) 启动后端

```powershell
python app.py
```

默认地址：`http://127.0.0.1:5000`

- 健康检查：`GET /api/health`
- 前向推理：`POST /api/forward`

## 3) 启动前端

在项目根目录：

```powershell
npm start
```

当前前端 A 模式会优先调用后端 `/api/forward`，后端不可用时自动回退本地计算。
