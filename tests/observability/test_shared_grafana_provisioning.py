from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAFANA_ROOT = REPO_ROOT / "monitoring" / "grafana"
NALUS_DATASOURCE = GRAFANA_ROOT / "provisioning" / "datasources" / "prometheus.yml"
ETERNAL_WORLD_DATASOURCE = (
    GRAFANA_ROOT / "provisioning" / "datasources" / "eternal-world.yml"
)
NALUS_PROVIDER = GRAFANA_ROOT / "provisioning" / "dashboards" / "dashboards.yml"
ETERNAL_WORLD_PROVIDER = (
    GRAFANA_ROOT / "provisioning" / "dashboards" / "eternal-world.yml"
)
NALUS_DASHBOARD = GRAFANA_ROOT / "dashboards" / "legal_answer_eval_dashboard.json"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(_read(path))
    assert isinstance(payload, dict)
    return payload


def _walk_panels(panels: list[dict]):
    for panel in panels:
        yield panel
        nested = panel.get("panels")
        if isinstance(nested, list):
            yield from _walk_panels(nested)


def test_existing_nalus_datasource_and_dashboard_contract_are_preserved() -> None:
    datasources = _load_yaml(NALUS_DATASOURCE)["datasources"]
    assert datasources == [
        {
            "name": "Prometheus",
            "type": "prometheus",
            "uid": "prometheus",
            "access": "proxy",
            "url": "http://prometheus:9090",
            "isDefault": True,
            "editable": False,
        }
    ]

    dashboard = json.loads(_read(NALUS_DASHBOARD))
    assert dashboard["uid"] == "nalus-legal-answer-eval"
    assert dashboard["title"] == "NALUS — Legal Answer Eval Metrics"
    for panel in _walk_panels(dashboard.get("panels", [])):
        datasource_ref = panel.get("datasource")
        if datasource_ref is not None:
            assert datasource_ref == {"type": "prometheus", "uid": "prometheus"}
        for target in panel.get("targets", []):
            target_datasource = target.get("datasource")
            if target_datasource is not None:
                assert target_datasource == {"type": "prometheus", "uid": "prometheus"}


def test_eternal_world_datasource_is_explicit_non_default_and_configurable() -> None:
    nalus = _load_yaml(NALUS_DATASOURCE)["datasources"]
    eternal_world = _load_yaml(ETERNAL_WORLD_DATASOURCE)["datasources"]
    assert eternal_world == [
        {
            "name": "Eternal World Prometheus",
            "type": "prometheus",
            "uid": "eternal-world-prometheus",
            "access": "proxy",
            "url": "$ETERNAL_WORLD_PROMETHEUS_URL",
            "isDefault": False,
            "editable": False,
        }
    ]

    combined = [*nalus, *eternal_world]
    assert len({item["uid"] for item in combined}) == len(combined)
    assert len({item["name"] for item in combined}) == len(combined)
    assert sum(bool(item["isDefault"]) for item in combined) == 1


def test_dashboard_providers_use_non_overlapping_project_folders() -> None:
    nalus_providers = _load_yaml(NALUS_PROVIDER)["providers"]
    eternal_world_providers = _load_yaml(ETERNAL_WORLD_PROVIDER)["providers"]
    assert len(nalus_providers) == 1
    assert len(eternal_world_providers) == 1

    nalus = nalus_providers[0]
    eternal_world = eternal_world_providers[0]
    assert nalus["name"] == "NALUS Legal Answer Eval"
    assert nalus["folder"] == "NALUS"
    assert nalus["options"]["path"] == "/var/lib/grafana/dashboards/nalus"
    assert eternal_world["name"] == "Eternal World Dashboards"
    assert eternal_world["folder"] == "Eternal World"
    assert eternal_world["options"]["path"] == "/var/lib/grafana/dashboards/eternal-world"
    assert eternal_world["allowUiUpdates"] is False

    combined = [nalus, eternal_world]
    assert len({item["name"] for item in combined}) == len(combined)
    assert len({item["folder"] for item in combined}) == len(combined)
    assert len({item["options"]["path"] for item in combined}) == len(combined)


def test_compose_mounts_both_dashboard_sources_read_only() -> None:
    grafana = _load_yaml(REPO_ROOT / "docker-compose.yml")["services"]["grafana"]

    assert grafana["ports"] == ["3002:3000"]
    assert "ETERNAL_WORLD_PROMETHEUS_URL=${ETERNAL_WORLD_PROMETHEUS_URL:-http://host.docker.internal:9090}" in grafana["environment"]
    assert grafana["extra_hosts"] == ["host.docker.internal:host-gateway"]
    assert "prometheus" in grafana["depends_on"]

    long_mounts = {
        item["target"]: item
        for item in grafana["volumes"]
        if isinstance(item, dict)
    }
    nalus_mount = long_mounts["/var/lib/grafana/dashboards/nalus"]
    assert nalus_mount["source"] == "./monitoring/grafana/dashboards"
    assert nalus_mount["read_only"] is True
    assert nalus_mount["bind"]["create_host_path"] is False

    eternal_world_mount = long_mounts["/var/lib/grafana/dashboards/eternal-world"]
    assert eternal_world_mount["source"] == "${ETERNAL_WORLD_DASHBOARD_DIR:-../eternal-world/monitoring/grafana/dashboards}"
    assert eternal_world_mount["read_only"] is True
    assert eternal_world_mount["bind"]["create_host_path"] is False

    provider_paths = {
        _load_yaml(NALUS_PROVIDER)["providers"][0]["options"]["path"],
        _load_yaml(ETERNAL_WORLD_PROVIDER)["providers"][0]["options"]["path"],
    }
    assert provider_paths == set(long_mounts)

    env_example = _read(REPO_ROOT / ".env.example")
    assert "ETERNAL_WORLD_PROMETHEUS_URL=http://host.docker.internal:9090" in env_example
    assert "ETERNAL_WORLD_DASHBOARD_DIR=../eternal-world/monitoring/grafana/dashboards" in env_example
