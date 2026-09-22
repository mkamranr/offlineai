"""Section 33: build-time versus runtime external dependencies.

The distinction is the whole value. A model fetched from huggingface.co while
building is fine - that is what a builder is for. The same URL in a service's
runtime configuration means the workload will reach out on a host that has no
route anywhere.

Equally important: the audit must not cry wolf on correct internal
configuration, because one that does is one operators learn to ignore.
"""

from __future__ import annotations

import pytest

from offlineai.schema.package import Package
from offlineai.security.network import Phase, audit_package

BASE = {
    "apiVersion": "offlineai/v1",
    "kind": "Package",
    "metadata": {"name": "audited", "version": "1.0.0"},
}


def package(**overrides: object) -> Package:
    return Package.model_validate({**BASE, **overrides})


def phases_for(audit: object, url_fragment: str) -> list[Phase]:
    return [f.phase for f in audit.findings if url_fragment in f.url]  # type: ignore[attr-defined]


class TestBuildTimeReferences:
    def test_a_model_repository_is_build_time(self) -> None:
        audit = audit_package(
            package(models=[{"name": "m", "source": {"type": "huggingface", "repo": "org/model"}}])
        )
        assert phases_for(audit, "org/model") == [Phase.BUILD]
        assert audit.clean

    def test_a_container_image_is_build_time(self) -> None:
        audit = audit_package(package(containers=[{"name": "c", "image": "redis:7"}]))
        assert phases_for(audit, "redis") == [Phase.BUILD]
        assert audit.clean

    def test_python_requirements_are_build_time(self) -> None:
        audit = audit_package(package(python={"requirements": ["requirements.txt"]}))
        assert phases_for(audit, "pypi.org") == [Phase.BUILD]
        assert audit.clean


class TestRuntimeDependencies:
    """The defect the audit exists to surface."""

    def test_an_external_api_endpoint_is_flagged(self) -> None:
        audit = audit_package(
            package(environment={"MODEL_API_ENDPOINT": "https://api.openai.com/v1"})
        )
        assert phases_for(audit, "api.openai.com") == [Phase.RUNTIME]
        assert not audit.clean

    def test_a_telemetry_url_is_flagged(self) -> None:
        audit = audit_package(
            package(environment={"TELEMETRY_URL": "https://telemetry.example.com/i"})
        )
        assert not audit.clean

    def test_a_build_time_host_in_runtime_config_is_flagged(self) -> None:
        """huggingface.co at build time is fine; as a runtime endpoint it is
        exactly the failure this catches."""
        audit = audit_package(package(environment={"HF_ENDPOINT": "https://huggingface.co"}))
        assert phases_for(audit, "huggingface.co") == [Phase.RUNTIME]
        assert not audit.clean

    def test_a_url_in_a_service_command_is_examined(self) -> None:
        audit = audit_package(
            package(
                containers=[{"name": "c", "image": "redis:7"}],
                services=[
                    {
                        "name": "s",
                        "container": "c",
                        "command": ["--webhook", "https://hooks.example.net/x"],
                    }
                ],
            )
        )
        assert any(f.phase in (Phase.RUNTIME, Phase.UNKNOWN) for f in audit.findings)
        assert not audit.clean

    def test_an_unclassifiable_url_is_surfaced_not_dropped(self) -> None:
        """A URL the audit quietly dropped would be worse than one it could not
        categorise."""
        audit = audit_package(package(environment={"SOMETHING": "https://unknown.example/x"}))
        assert phases_for(audit, "unknown.example") == [Phase.UNKNOWN]
        assert not audit.clean


class TestInternalReferencesAreNotFalsePositives:
    """An audit that flags correct internal configuration gets ignored."""

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:6379",
            "http://127.0.0.1:8000",
            "http://10.0.1.5:8000",
            "http://192.168.1.10:8000",
            "http://172.16.0.5:8000",
        ],
    )
    def test_loopback_and_private_ranges_are_local(self, url: str) -> None:
        audit = audit_package(package(environment={"SERVICE_URL": url}))
        assert phases_for(audit, url.split("//")[1].split(":")[0]) == [Phase.LOCAL]
        assert audit.clean

    def test_a_declared_service_name_is_local(self) -> None:
        """This is the rag-stack case: http://vllm:8000 where vllm is a service
        in the same package."""
        audit = audit_package(
            package(
                containers=[{"name": "vllm", "image": "vllm/vllm-openai:v0.6.3"}],
                services=[{"name": "vllm", "container": "vllm"}],
                environment={"LLM_BASE_URL": "http://vllm:8000/v1"},
            )
        )
        assert phases_for(audit, "//vllm:8000") == [Phase.LOCAL]
        assert audit.clean

    def test_any_single_label_hostname_is_local(self) -> None:
        """A name with no dot cannot be public DNS; it is a container on the
        deployment network."""
        audit = audit_package(package(environment={"CACHE_URL": "http://redis:6379"}))
        assert phases_for(audit, "//redis") == [Phase.LOCAL]
        assert audit.clean

    def test_the_detail_names_the_service_it_resolves_to(self) -> None:
        audit = audit_package(
            package(
                containers=[{"name": "qdrant", "image": "qdrant/qdrant:v1"}],
                services=[{"name": "qdrant", "container": "qdrant"}],
                environment={"QDRANT_URL": "http://qdrant:6333"},
            )
        )
        finding = next(f for f in audit.findings if "6333" in f.url)
        assert finding.detail is not None and "qdrant" in finding.detail


class TestRealExamples:
    """The shipped examples must audit cleanly, or the examples are wrong."""

    @pytest.mark.parametrize(
        "example", ["hello-ai", "simple-python", "whisper", "qwen-vllm", "rag-stack"]
    )
    def test_shipped_examples_declare_no_runtime_dependency(self, example: str) -> None:
        from pathlib import Path

        from offlineai.resolver.package import load_package

        root = Path(__file__).resolve().parents[2] / "examples" / example
        pkg, _, _ = load_package(root)
        audit = audit_package(pkg)
        assert audit.clean, f"{example} would reach the network at run time: " + ", ".join(
            f.url for f in audit.runtime_dependencies
        )
