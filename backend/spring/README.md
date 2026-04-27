# DeepVision Studio Spring Backend

Spring 后端负责用户注册登录、JWT 鉴权、A 模式前向传播记录的保存/查询/删除，以及统一代理 Python 前向传播服务。
同时提供 B 模式训练数据集导入、训练任务模拟、训练状态控制和 WebSocket 指标推送接口。

## 技术栈

- Java 17+
- Spring Boot 3.3
- Spring Web
- Spring Security
- Spring Data JPA
- H2 文件数据库
- JWT

## 环境变量

开发环境可以不配置环境变量，默认值在 `src/main/resources/application.yml` 中。

```powershell
$env:DEEPVISION_JWT_SECRET="replace-with-a-random-secret-at-least-32-bytes"
$env:DEEPVISION_FORWARD_BASE_URL="http://127.0.0.1:5000"
```

可选配置：

```powershell
$env:DEEPVISION_DB_URL="jdbc:h2:file:./data/deepvision;AUTO_SERVER=TRUE"
$env:DEEPVISION_DB_USERNAME="sa"
$env:DEEPVISION_DB_PASSWORD=""
$env:DEEPVISION_UPLOAD_ROOT="./uploads"
$env:DEEPVISION_DATASET_ROOT="./datasets"
$env:DEEPVISION_JWT_EXPIRATION_MINUTES="10080"
$env:DEEPVISION_TRAINING_STREAM_BASE_URL="ws://127.0.0.1:8080"
$env:DEEPVISION_TRAINING_PYTHON="C:/Users/lizihan/miniconda3/envs/dl-platform/python.exe"
$env:DEEPVISION_TRAINING_WORKER_SCRIPT="../python-training/training_worker.py"
```

## 启动

```powershell
cd backend/spring
mvn spring-boot:run
```

默认地址：

- Spring 后端：`http://127.0.0.1:8080`
- H2 控制台：`http://127.0.0.1:8080/h2-console`
- 上传图片访问：`http://127.0.0.1:8080/uploads/...`

## 接口

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/forward`
- `GET /api/forward/health`
- `GET /api/a/forward-records`
- `POST /api/a/forward-records`
- `GET /api/a/forward-records/{id}`
- `DELETE /api/a/forward-records/{id}`
- `GET /api/training/datasets/builtin`
- `GET /api/training/datasets/{datasetId}`
- `POST /api/training/datasets/imports`
- `POST /api/training/start`
- `WS /api/training/stream?jobId={jobId}`
- `GET /api/training/{jobId}/status`
- `GET /api/training/{jobId}/weights/histogram`
- `POST /api/training/{jobId}/pause`
- `POST /api/training/{jobId}/resume`
- `POST /api/training/{jobId}/stop`
- `POST /api/training/{jobId}/reset`
- `POST /api/training/{jobId}/save`

`/api/a/forward-records` 需要 `Authorization: Bearer <token>`。
`/api/training/**` 用于前端教学演示联调，默认不要求登录。
