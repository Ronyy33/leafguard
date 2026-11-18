# AI Disclosure: LeafGuard

**Rohan Sirsewad (DA24B023)** · DA3408 AI Operations · IIT Madras

## 1. Tools used

| Tool | Used for |
|---|---|
| Claude (Anthropic) | Explaining concepts, suggesting implementation approaches, generating code boilerplate, assisting with debugging, reviewing code and proofreading documentation. |


## 2. How it was used

| Area | What I did | What Claude did |
|---|---|---|
| Planning | Defined the project topic (plant disease detection for farmers), the 10 crop/disease classes, the success metrics and the day-by-day implementation plan. | Helped structure the development schedule. |
| Architecture | Designed the loosely coupled architecture (UI, API, model server, trainer, monitoring), selected the DA3408 toolchain and made the trade-offs recorded in `docs/HLD.md` (Ray vs Spark, MobileNetV2 vs ResNet18, z-score drift vs KS/PSI, SQLite vs PostgreSQL). | Explained alternative approaches and helped evaluate design decisions. |
| Data pipeline | Curated the PlantVillage subset and implemented the Ray Data ingest, validate and preprocess stages, the EDA stage, the drift baseline statistics, the feedback import and the dataset download script, all wired into the DVC DAG. | Assisted with Ray Data and DVC syntax and with pipeline debugging. |
| Model | Implemented MobileNetV2 transfer learning, the evaluation and promotion gate, and field-condition (dark/blur) augmentation. Ran and compared the experiments (frozen vs fine-tuned backbone, learning rate, robustness augmentation) that led to production model v5. | Helped with MLflow syntax, experiment tracking, model registration, promotion and rollback, and the Grad-CAM implementation. |
| Backend and UI | Developed the FastAPI backend (prediction, Grad-CAM, drift monitoring, feedback, encrypted storage, pipeline control), the trainer service and the HTML/JS frontend served by nginx. | Generated initial boilerplate and helped with API implementation and UI integration. |
| Docker and Compose | Wrote the multi-stage Dockerfile and the frontend Dockerfile, configured the Compose services, volumes, healthchecks and networking. | Provided a starting Docker Compose configuration and helped resolve container-related issues. |
| Monitoring and retraining | Configured Prometheus metrics and alert rules, Alertmanager routing and the Grafana dashboard, and implemented drift- and accuracy-triggered automatic retraining. | Assisted with Prometheus rules, PromQL syntax, Alertmanager webhook configuration and the Grafana dashboard JSON. |
| Debugging | Investigated errors, examined logs and implemented fixes. | Helped identify root causes and suggested fixes for issues such as the augmentation deadlock in DataLoader workers, the Alpine frontend healthcheck resolving `localhost` to IPv6, container user-ID permissions in the trainer and ANSI codes in the streamed pipeline log. |
| Testing | Designed test cases and executed unit, integration and end-to-end tests, including the DVC test stage and the Docker end-to-end retrain loop. | Helped with pytest fixtures, test boilerplate and debugging test failures. |
| Documentation | Prepared the problem statement, architecture, HLD, LLD, test plan, user manual and demo script. | Assisted with diagrams, proofreading and improving clarity. |
| Cleanup | Reviewed the codebase, ran flake8/black and simplified components to align with course requirements. | Suggested simplifications and helped improve code readability. |

## 3. What I did

- Defined the project topic, scope and objectives in `docs/problem_statement.md`.
- Designed the architecture and selected the technology stack.
- Implemented the core data pipeline, model workflow, backend, trainer and frontend.
- Configured Docker, the DVC pipeline and the monitoring infrastructure.
- Designed and executed the experiments and tests.
- Investigated errors, verified fixes and integrated the different components.
- Installed and configured the development environment.
- Reviewed and finalized the project documentation.

## 4. Impact

Claude helped accelerate development by providing implementation guidance, code boilerplate, debugging assistance and documentation feedback. The system design, architectural decisions, core implementation, experiments and verification were my responsibility.
