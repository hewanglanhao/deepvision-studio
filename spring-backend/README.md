# DeepVision Studio Spring Backend

这个后端负责全站用户注册登录，以及 A 模式前向传播记录的保存、查询、回溯和删除。
现有 Python Flask 后端仍然只负责 A 模式前向传播计算，默认端口还是 `5000`。

## 技术栈

- Java 17+
- Spring Boot 3.3
- Spring Web
- Spring Security
- Spring Data JPA
- H2 文件数据库
- JWT

## 环境变量

开发环境可以不配置环境变量，后端会使用 `application.yml` 里的默认值。
正式提交或演示时建议至少配置 JWT 密钥：

```powershell
$env:DEEPVISION_JWT_SECRET="replace-with-a-random-secret-at-least-32-bytes"
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
cd spring-backend
mvn clean package
java -jar target/studio-backend-0.0.1-SNAPSHOT.jar
```

默认地址：

- Spring 后端：`http://127.0.0.1:8080`
- H2 控制台：`http://127.0.0.1:8080/h2-console`
- 图片访问：`http://127.0.0.1:8080/uploads/...`

H2 控制台默认连接信息：

- JDBC URL：`jdbc:h2:file:./data/deepvision;AUTO_SERVER=TRUE`
- User Name：`sa`
- Password：空

## 接口

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `GET /api/a/forward-records`
- `POST /api/a/forward-records`
- `GET /api/a/forward-records/{id}`
- `DELETE /api/a/forward-records/{id}`

`/api/a/forward-records` 需要 `Authorization: Bearer <token>`。

## 与前端、Python 后端一起运行

分别开三个终端：

```powershell
# 1. Angular
npm start

# 2. Python Flask A 模式计算后端
cd backend
python app.py

# 3. Spring 登录与记录后端
cd spring-backend
mvn clean package
java -jar target/studio-backend-0.0.1-SNAPSHOT.jar
```
