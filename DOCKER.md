# DeepVision Studio Docker Deployment

## Local Run

```powershell
docker compose up --build
```

Open:

- Frontend: `http://localhost:4200`
- Spring health check through nginx: `http://localhost:4200/api/health`

The compose stack contains:

- `frontend`: Angular production build served by nginx.
- `spring-backend`: Spring Boot backend. The Python training worker is installed inside this image because Spring launches it as a subprocess.
- `python-forward`: Flask forward-pass service, called only by Spring through the Docker network.

## Useful Commands

```powershell
docker compose ps
docker compose logs -f
docker compose down
docker compose down -v
```

`docker compose down -v` removes the H2 database, uploads, datasets, and training job volumes.

## Docker Hub / Cloud Server Flow

Build and tag:

```powershell
docker build -t your-dockerhub-name/deepvision-frontend:latest ./frontend
docker build -t your-dockerhub-name/deepvision-spring-backend:latest -f backend/spring/Dockerfile .
docker build -t your-dockerhub-name/deepvision-python-forward:latest ./backend/python-forward
```

Push:

```powershell
docker push your-dockerhub-name/deepvision-frontend:latest
docker push your-dockerhub-name/deepvision-spring-backend:latest
docker push your-dockerhub-name/deepvision-python-forward:latest
```

On the server, you can use the same `docker-compose.yml`, replace the three `image:` values with your Docker Hub image names, and run:

```bash
docker compose pull
docker compose up -d
```

For a public cloud host, map frontend port `80:80` instead of `4200:80`, open the security group/firewall port, and put secrets such as `DEEPVISION_JWT_SECRET` and `ARK_API_KEY` in a server-side `.env` file.
