# Docker Upload

Ovaj repo vec ima lokalni `Dockerfile` i dev `docker-compose.yml`. Za push na Docker Hub i deploy iz registry-ja koristi sledece:

## Lokalni push

Prijavi se:

```powershell
docker login
```

Objavi API i exporter image:

```powershell
.\scripts\docker_publish.ps1 -DockerHubNamespace your-dockerhub-user -Tag v1.0.0 -AlsoTagLatest
```

Samo API image:

```powershell
.\scripts\docker_publish.ps1 -DockerHubNamespace your-dockerhub-user -Tag v1.0.0 -SkipExporter
```

Ako vec imas lokalno izgradjene compose imageove i ne zelis novi build od nule:

```powershell
.\scripts\docker_publish.ps1 -DockerHubNamespace your-dockerhub-user -Tag v1.0.0 -UseExistingImages -AlsoTagLatest
```

Default lokalni source imageovi za ovaj fallback su:

- `nalus-scraper-api:latest`
- `nalus-scraper-nalus-eval-metrics-exporter:latest`

Mozes i preko env varijabli:

```powershell
$env:DOCKERHUB_NAMESPACE="your-dockerhub-user"
$env:IMAGE_TAG="v1.0.0"
.\scripts\docker_publish.ps1 -AlsoTagLatest
```

## GitHub Actions publish

Workflow: [../.github/workflows/docker-publish.yml](../.github/workflows/docker-publish.yml)

Potrebno je dodati:

- secret `DOCKERHUB_USERNAME`
- secret `DOCKERHUB_TOKEN`
- optional repo variable `DOCKERHUB_NAMESPACE`

Trigger opcije:

- manualno preko `workflow_dispatch`
- automatski kada pushas git tag oblika `docker-v1.0.0`

## Deploy iz Docker Hub-a

Registry compose fajl: [../docker-compose.registry.yml](../docker-compose.registry.yml)

Primjer:

```powershell
$env:DOCKERHUB_NAMESPACE="your-dockerhub-user"
$env:IMAGE_TAG="v1.0.0"
docker compose -f docker-compose.registry.yml up -d
```

Ako model, storage ili batches nisu u default lokalnim folderima, postavi:

- `NALUS_MODELS_HOST_DIR`
- `NALUS_STORAGE_HOST_DIR`
- `NALUS_BATCHES_HOST_DIR`
- `NALUS_ARTIFACTS_HOST_DIR`
- `NALUS_APP_ARTIFACTS_HOST_DIR`
