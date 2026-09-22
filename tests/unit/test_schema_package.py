"""Section 8: the offlineai.yaml package definition.

The schema is versioned and strict. Rejecting unknown keys matters more than it
might look: a typo like `containters:` that validates silently produces a bundle
missing an image, and the operator only finds out on the air-gapped side where
they cannot fix it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from offlineai.schema.package import (
    API_VERSION,
    ContainerSpec,
    ImageReference,
    Package,
    ServiceSpec,
)

MINIMAL = {
    "apiVersion": API_VERSION,
    "kind": "Package",
    "metadata": {"name": "hello-ai", "version": "1.0.0"},
}


def pkg(**overrides: object) -> dict[str, object]:
    return {**MINIMAL, **overrides}


class TestEnvelope:
    def test_minimal_package_validates(self) -> None:
        package = Package.model_validate(MINIMAL)
        assert package.metadata.name == "hello-ai"
        assert package.metadata.version == "1.0.0"

    def test_api_version_must_match(self) -> None:
        with pytest.raises(ValidationError, match="apiVersion"):
            Package.model_validate(pkg(apiVersion="offlineai/v2"))

    def test_kind_must_be_package(self) -> None:
        with pytest.raises(ValidationError, match="kind"):
            Package.model_validate(pkg(kind="Deployment"))

    def test_unknown_top_level_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="containters"):
            Package.model_validate(pkg(containters=[]))

    def test_unknown_nested_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="verison"):
            Package.model_validate(
                pkg(metadata={"name": "x", "version": "1.0.0", "verison": "typo"})
            )


class TestMetadata:
    @pytest.mark.parametrize("name", ["qwen3-30b", "hello-ai", "rag-stack", "a", "a1-b2"])
    def test_valid_names(self, name: str) -> None:
        Package.model_validate(pkg(metadata={"name": name, "version": "1.0.0"}))

    @pytest.mark.parametrize(
        "name",
        ["Qwen3", "has space", "trailing-", "-leading", "under_score", "", "a" * 129, "../x"],
    )
    def test_invalid_names(self, name: str) -> None:
        with pytest.raises(ValidationError):
            Package.model_validate(pkg(metadata={"name": name, "version": "1.0.0"}))

    def test_name_is_used_as_a_filename_so_must_stay_path_safe(self) -> None:
        """Bundle files are named <name>-<version>.offlineai; a name containing
        a separator would write outside the output directory."""
        with pytest.raises(ValidationError):
            Package.model_validate(pkg(metadata={"name": "a/b", "version": "1.0.0"}))

    @pytest.mark.parametrize("version", ["1.0.0", "0.1.0", "2.1.3-rc1", "1.0.0+build5"])
    def test_valid_versions(self, version: str) -> None:
        Package.model_validate(pkg(metadata={"name": "x", "version": version}))

    @pytest.mark.parametrize("version", ["", "not a version", "1.0.0/../x"])
    def test_invalid_versions(self, version: str) -> None:
        with pytest.raises(ValidationError):
            Package.model_validate(pkg(metadata={"name": "x", "version": version}))


class TestImageReference:
    @pytest.mark.parametrize(
        ("raw", "repository", "tag"),
        [
            ("redis:7", "redis", "7"),
            ("nginx:alpine", "nginx", "alpine"),
            ("vllm/vllm-openai:latest", "vllm/vllm-openai", "latest"),
            ("ghcr.io/org/app:1.2", "ghcr.io/org/app", "1.2"),
            ("redis", "redis", "latest"),
            ("localhost:5000/app:v1", "localhost:5000/app", "v1"),
        ],
    )
    def test_parses_tag_forms(self, raw: str, repository: str, tag: str) -> None:
        ref = ImageReference.parse(raw)
        assert ref.repository == repository
        assert ref.tag == tag

    def test_parses_digest_form(self) -> None:
        digest = "sha256:" + "a" * 64
        ref = ImageReference.parse(f"redis@{digest}")
        assert ref.repository == "redis"
        assert ref.digest == digest

    def test_registry_port_is_not_mistaken_for_a_tag(self) -> None:
        ref = ImageReference.parse("registry.internal:5000/team/app")
        assert ref.repository == "registry.internal:5000/team/app"
        assert ref.tag == "latest"

    def test_round_trips(self) -> None:
        assert str(ImageReference.parse("vllm/vllm-openai:latest")) == "vllm/vllm-openai:latest"


class TestContainers:
    def test_image_is_parsed(self) -> None:
        container = ContainerSpec.model_validate({"name": "vllm", "image": "redis:7"})
        assert container.reference.repository == "redis"

    def test_digest_must_be_well_formed(self) -> None:
        with pytest.raises(ValidationError):
            ContainerSpec.model_validate(
                {"name": "vllm", "image": "redis:7", "digest": "sha256:short"}
            )

    def test_valid_digest_is_accepted(self) -> None:
        container = ContainerSpec.model_validate(
            {"name": "vllm", "image": "redis:7", "digest": "sha256:" + "b" * 64}
        )
        assert container.digest is not None

    def test_duplicate_container_names_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            Package.model_validate(
                pkg(
                    containers=[
                        {"name": "app", "image": "redis:7"},
                        {"name": "app", "image": "nginx:alpine"},
                    ]
                )
            )


class TestServices:
    def test_service_must_reference_a_declared_container(self) -> None:
        with pytest.raises(ValidationError, match="ghost"):
            Package.model_validate(
                pkg(
                    containers=[{"name": "app", "image": "redis:7"}],
                    services=[{"name": "api", "container": "ghost"}],
                )
            )

    def test_service_referencing_a_declared_container_is_accepted(self) -> None:
        package = Package.model_validate(
            pkg(
                containers=[{"name": "app", "image": "redis:7"}],
                services=[{"name": "api", "container": "app", "ports": ["8000:8000"]}],
            )
        )
        assert package.services[0].container == "app"

    @pytest.mark.parametrize("port", ["8000:8000", "127.0.0.1:8000:8000", "8000:8000/udp"])
    def test_valid_port_mappings(self, port: str) -> None:
        ServiceSpec.model_validate({"name": "api", "container": "app", "ports": [port]})

    @pytest.mark.parametrize("port", ["8000", "abc:8000", "8000:99999", "-1:8000", ""])
    def test_invalid_port_mappings(self, port: str) -> None:
        with pytest.raises(ValidationError):
            ServiceSpec.model_validate({"name": "api", "container": "app", "ports": [port]})

    def test_host_port_is_exposed_for_endpoint_reporting(self) -> None:
        service = ServiceSpec.model_validate(
            {"name": "api", "container": "app", "ports": ["8001:8000"]}
        )
        assert service.host_ports() == [8001]


class TestModels:
    def test_huggingface_source(self) -> None:
        package = Package.model_validate(
            pkg(
                models=[
                    {
                        "name": "qwen3",
                        "source": {"type": "huggingface", "repo": "Qwen/Qwen3-30B"},
                        "destination": "/models/qwen3",
                    }
                ]
            )
        )
        assert package.models[0].source.type == "huggingface"
        assert package.models[0].source.repo == "Qwen/Qwen3-30B"

    def test_unknown_source_type_is_rejected_with_the_known_list(self) -> None:
        with pytest.raises(ValidationError, match="huggingface"):
            Package.model_validate(
                pkg(models=[{"name": "m", "source": {"type": "telepathy", "repo": "x"}}])
            )

    def test_huggingface_source_requires_a_repo(self) -> None:
        with pytest.raises(ValidationError):
            Package.model_validate(pkg(models=[{"name": "m", "source": {"type": "huggingface"}}]))

    def test_duplicate_model_names_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            Package.model_validate(
                pkg(
                    models=[
                        {"name": "m", "source": {"type": "huggingface", "repo": "a/b"}},
                        {"name": "m", "source": {"type": "huggingface", "repo": "c/d"}},
                    ]
                )
            )


class TestHardware:
    def test_gpu_requirements(self) -> None:
        package = Package.model_validate(
            pkg(hardware={"gpu": {"required": True, "vendor": "nvidia", "minimum_vram_gb": 48}})
        )
        assert package.hardware.gpu is not None
        assert package.hardware.gpu.minimum_vram_gb == 48

    def test_negative_vram_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Package.model_validate(pkg(hardware={"gpu": {"minimum_vram_gb": -1}}))

    def test_defaults_to_no_gpu_requirement(self) -> None:
        assert Package.model_validate(MINIMAL).hardware.gpu is None


class TestSecrets:
    def test_only_names_are_carried_never_values(self) -> None:
        """Section 40: the bundle contains the name, not the value."""
        package = Package.model_validate(
            pkg(secrets={"external": ["HUGGINGFACE_TOKEN", "DATABASE_PASSWORD"]})
        )
        assert package.secrets.external == ["HUGGINGFACE_TOKEN", "DATABASE_PASSWORD"]
        assert "value" not in package.secrets.model_dump()

    def test_a_secret_entry_carrying_a_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Package.model_validate(
                pkg(secrets={"external": [{"name": "TOKEN", "value": "hunter2"}]})
            )


class TestEnvironment:
    def test_values_must_be_strings_to_match_docker_semantics(self) -> None:
        package = Package.model_validate(pkg(environment={"HF_HUB_OFFLINE": "1"}))
        assert package.environment["HF_HUB_OFFLINE"] == "1"

    def test_numeric_values_are_coerced_to_strings(self) -> None:
        package = Package.model_validate(pkg(environment={"WORKERS": 4}))
        assert package.environment["WORKERS"] == "4"


class TestBundleNaming:
    def test_bundle_filename_matches_the_documented_convention(self) -> None:
        package = Package.model_validate(pkg(metadata={"name": "qwen3-30b", "version": "1.0.0"}))
        assert package.bundle_filename() == "qwen3-30b-1.0.0.offlineai"
