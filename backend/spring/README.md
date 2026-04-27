# DeepVision Studio Spring Backend

Spring 后端负责用户注册登录、JWT 鉴权、A 模式前向传播记录的保存/查询/删除，以及统一代理 Python 前向传播服务。

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
$env:DEEPVISION_JWT_EXPIRATION_MINUTES="10080"
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

`/api/a/forward-records` 需要 `Authorization: Bearer <token>`。
