# Backend Structure

The backend is split by runtime boundary first.

## spring

Spring Boot is the application API gateway and persistence layer.

- `auth`: users, JWT, and authentication endpoints.
- `common`: cross-cutting web/security/error handling.
- `forward`: proxy endpoint for the Python forward-pass runtime plus Mode A forward-pass saved records and preview-image storage.
- `training`: datasets, training jobs, checkpoint persistence, streams, and collaboration sockets.
- `llm`: assistant chat proxy.

This package split follows the UI domain boundaries. Mode A's saved forward snapshots live in `forward`; Mode B's persisted training history lives in `training` as checkpoints because those records are tied to checkpoint files, test runs, and training datasets. Future Java refactors should keep controllers, DTOs, services, and repositories inside the same domain package unless a class is genuinely cross-cutting.

## python-forward

Python runtime for real forward-pass execution and tensor visualization data.

## python-training

Python runtime for real model training, checkpoint generation, and evaluation.
