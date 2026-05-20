FROM maven:3.9.9-eclipse-temurin-17 AS build

WORKDIR /workspace/backend/spring
COPY backend/spring/pom.xml .
RUN mvn -B dependency:go-offline
COPY backend/spring/src ./src
RUN mvn -B package -DskipTests

FROM eclipse-temurin:17-jre-jammy

WORKDIR /app
RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

COPY backend/python-training /app/python-training
RUN python3 -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r /app/python-training/requirements.txt

COPY --from=build /workspace/backend/spring/target/*.jar /app/app.jar

ENV DEEPVISION_DB_URL=jdbc:h2:file:/app/data/deepvision \
    DEEPVISION_DB_USERNAME=sa \
    DEEPVISION_DB_PASSWORD= \
    DEEPVISION_UPLOAD_ROOT=/app/uploads \
    DEEPVISION_DATASET_ROOT=/app/datasets \
    DEEPVISION_TRAINING_JOBS_ROOT=/app/training-jobs \
    DEEPVISION_TRAINING_PYTHON=python3 \
    DEEPVISION_TRAINING_WORKER_SCRIPT=/app/python-training/training_worker.py

EXPOSE 8080
ENTRYPOINT ["java", "-jar", "/app/app.jar"]
