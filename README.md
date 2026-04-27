# DeepVision Studio

深度学习算法可视化仿真平台，当前拆分为前端和后端两块：

- `frontend/`: Angular 前端应用。
- `backend/spring/`: Spring Boot 业务后端，负责登录注册、JWT、A 模式记录保存，以及转发前向传播计算请求。
- `backend/python-forward/`: Python Flask 推理服务，只负责 A 模式前向传播计算。

## 本地启动

分别打开三个终端：

```powershell
# 1. 前端
cd frontend
npm install
npm start
```

```powershell
# 2. Spring 业务后端
cd backend/spring
mvn spring-boot:run
```

```powershell
# 3. Python 前向传播服务
cd backend/python-forward
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

默认访问地址：

- 前端：`http://localhost:4200/`
- Spring 后端：`http://127.0.0.1:8080`
- Python 推理服务：`http://127.0.0.1:5000`

前端只需要调用 Spring 的 `8080` 端口；Spring 会把 `POST /api/forward` 转发给 Python 服务。

## 构建与验证

```powershell
cd frontend
npm run build
```

```powershell
cd backend/spring
mvn test
```

## 目录说明

- `frontend/src/app/app.ts`: 页面状态与主要交互逻辑。
- `frontend/src/app/app.html`: 主界面结构。
- `frontend/src/app/app.css`: 页面样式与响应式布局。
- `frontend/src/app/sim-models.ts`: 统一类型定义。
- `frontend/src/app/sim-engine.ts`: 本地仿真、训练曲线、视觉数据生成和模板逻辑。
- `backend/spring/src/main/java`: Spring Boot 后端源码。
- `backend/python-forward/forward_engine.py`: Python 前向传播计算核心。
