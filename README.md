# DeepVision Studio

DeepVision Studio 是一个面向深度学习教学的可视化实验平台，包含前向传播实验、模型训练、CNN/Transformer 解释、反向传播可视化、RNN/BPTT 演示、AI 辅助问答、在线协作和 Docker 部署。

完整开发文档见 [docs/development-document.md](docs/development-document.md)。该文档按项目架构、核心能力、成员分工、实现难点和部署方案组织，是当前项目的主要 README。

## 核心模块

| 模块 | 说明 |
| --- | --- |
| `frontend/` | Angular 前端应用，承载 A/B/C/D/E/F 模式、AI 博物馆、教学文档和共享浮标 |
| `backend/spring/` | Spring Boot 业务后端，负责认证、JWT、H2、forward 代理、训练任务、checkpoint、LLM 代理和 WebSocket |
| `backend/python-forward/` | Python Flask + NumPy 前向传播服务，服务 A 模式 |
| `backend/python-training/` | PyTorch 训练 worker，服务 B 模式训练、测试、checkpoint 和单样本推理 |
| `docs/development-document.md` | 项目开发文档与答辩维护主文档 |

## 本地启动

分别打开终端启动前端、Spring 后端和 Python forward 服务：

```powershell
cd frontend
npm install
npm start
```

```powershell
cd backend/spring
mvn spring-boot:run
```

```powershell
cd backend/python-forward
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

默认访问地址：

| 服务 | 地址 |
| --- | --- |
| 前端 | `http://localhost:4200/` |
| Spring 后端 | `http://127.0.0.1:8080` |
| Python forward | `http://127.0.0.1:5000` |

前端只需要访问 Spring 后端；Spring 会代理 `/api/forward` 到 Python forward 服务。B 模式训练 worker 由 Spring 按训练任务启动，不需要作为常驻 HTTP 服务单独运行。

## 构建与验证

```powershell
cd frontend
npm run build
```

```powershell
cd backend/spring
mvn test
```

## 文档约定

- [README.md](README.md)：项目入口说明。
- [docs/development-document.md](docs/development-document.md)：完整开发文档。
