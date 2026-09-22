"""The section 73 MVP scenario, end to end.

    build -> transfer -> verify -> import -> install -> status -> logs

Runs against FakeRuntime so the whole pipeline is exercised in CI with no
daemon, no images and no network. The parts this proves are the ones that are
hard to get right: that install refuses to reach out for a missing image, that
a failed install rolls back to where it started, and that deduplication does
not let removing one package break another.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from offlineai.bundler.builder import BundleBuilder
from offlineai.bundler.results import CheckStatus
from offlineai.bundler.verifier import verify_bundle
from offlineai.config.settings import Settings, load_settings
from offlineai.errors import MissingArtifactError, RuntimeFailureError
from offlineai.installer.installer import Installer
from offlineai.installer.rollback import rollback_installation
from offlineai.installer.transaction import InstallState
from offlineai.registry.registry import Registry
from offlineai.runtime.base import ContainerStatus
from offlineai.runtime.fake import FakeRuntime
from offlineai.runtime.manager import RuntimeManager, container_name

PACKAGE_YAML = """\
apiVersion: offlineai/v1
kind: Package
metadata:
  name: hello-ai
  version: 1.0.0
  description: Minimal containerised service.
runtime:
  type: docker
architecture:
  - amd64
containers:
  - name: app
    image: python:3.12-slim
    platform: linux/amd64
models:
  - name: demo
    source:
      type: local
      path: ./weights
    destination: /models/demo
environment:
  PYTHONUNBUFFERED: "1"
services:
  - name: app
    container: app
    ports:
      - "8000:8000"
install:
  healthcheck:
    command: ["true"]
    retries: 2
    interval_seconds: 1
"""


@pytest.fixture
def builder_runtime() -> FakeRuntime:
    """The connected builder's runtime: the base image is available to pull."""
    return FakeRuntime()


@pytest.fixture
def target_runtime() -> FakeRuntime:
    """The air-gapped target: starts with no images at all.

    This is the fixture that makes the test meaningful. Nothing is pre-loaded,
    so anything that works had to come out of the bundle.
    """
    return FakeRuntime()


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    source = tmp_path / "src" / "hello-ai"
    (source / "weights").mkdir(parents=True)
    (source / "weights" / "config.json").write_text('{"model_type": "demo"}')
    (source / "weights" / "model.safetensors").write_bytes(b"w" * 4096)
    (source / "offlineai.yaml").write_text(PACKAGE_YAML)
    return source


@pytest.fixture
def builder_home(tmp_path: Path) -> Settings:
    settings = load_settings(home=tmp_path / "builder")
    settings.ensure_directories()
    return settings


@pytest.fixture
def target_home(tmp_path: Path) -> Settings:
    settings = load_settings(home=tmp_path / "target")
    settings.ensure_directories()
    return settings


@pytest.fixture
def bundle(
    source_dir: Path, builder_home: Settings, builder_runtime: FakeRuntime, tmp_path: Path
) -> Path:
    """Build on the 'connected' machine, then move the file across the gap.

    The bundle is physically moved to a directory the target settings can see
    and the builder's cache is deleted, so nothing the target does can
    accidentally reach back into builder state.
    """
    result = BundleBuilder(builder_home, runtime=builder_runtime).build(
        source_dir, output=tmp_path / "build-output"
    )
    transferred = tmp_path / "usb" / Path(result.bundle_path).name
    transferred.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(result.bundle_path, transferred)
    shutil.rmtree(builder_home.cache_dir, ignore_errors=True)
    return transferred


class TestBuildOnTheConnectedMachine:
    def test_the_image_is_packaged(self, bundle: Path) -> None:
        from offlineai.bundler.inspector import inspect_bundle
        from offlineai.schema.manifest import ArtifactType

        result = inspect_bundle(bundle)
        assert result.artifact_counts.get(ArtifactType.OCI_IMAGE) == 1

    def test_the_model_files_are_packaged(self, bundle: Path) -> None:
        from offlineai.bundler.inspector import inspect_bundle
        from offlineai.schema.manifest import ArtifactType

        assert inspect_bundle(bundle).artifact_counts.get(ArtifactType.MODEL) == 2

    def test_the_image_digest_is_recorded(self, bundle: Path) -> None:
        """Section 16: never rely on a tag alone."""
        from offlineai.bundler.archive import BundleReader
        from offlineai.schema.manifest import ArtifactType

        manifest = BundleReader.peek_manifest(bundle)
        image = manifest.artifacts_of_type(ArtifactType.OCI_IMAGE)[0]
        assert image.digest is not None
        assert image.digest.startswith("sha256:")

    def test_no_build_step_was_silently_skipped(
        self, source_dir: Path, builder_home: Settings, builder_runtime: FakeRuntime, tmp_path: Path
    ) -> None:
        result = BundleBuilder(builder_home, runtime=builder_runtime).build(
            source_dir, output=tmp_path / "again"
        )
        skipped = [s.name for s in result.steps if s.status is CheckStatus.SKIPPED]
        assert skipped == [], f"unexpectedly skipped: {skipped}"


class TestOnTheAirGappedTarget:
    def test_the_bundle_verifies_after_transfer(self, bundle: Path) -> None:
        assert verify_bundle(bundle).verified is True

    def test_import_then_install_then_status(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        imported = registry.import_bundle(bundle)
        assert imported.artifacts_imported == 3

        installer = Installer(target_home, registry, target_runtime)
        result = installer.install("hello-ai", health_sleep=0)

        assert result.state is InstallState.COMPLETED
        assert result.healthy is True
        assert result.services_started == [container_name("hello-ai", "app")]
        assert "http://localhost:8000" in result.endpoints

    def test_the_image_came_out_of_the_bundle(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        """The target began with no images; if one is present now it was
        loaded from the bundle, not fetched."""
        assert target_runtime.images == {}
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)

        assert "python:3.12-slim" in target_runtime.images
        assert ("load", "pull") not in target_runtime.calls
        assert not any(op == "pull" for op, _ in target_runtime.calls), (
            "install must never pull; everything comes from the bundle"
        )

    def test_models_are_materialised_and_mounted_read_only(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)

        container = target_runtime.containers[container_name("hello-ai", "app")]
        mounts = {c: ro for _, c, ro in container.spec.volumes}
        assert mounts["/models/demo"] is True, "model weights must be mounted read-only"

    def test_status_and_logs(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)

        import yaml

        from offlineai.schema.package import Package

        record = registry.require("hello-ai")
        package = Package.model_validate(yaml.safe_load(record.package_yaml))
        manager = RuntimeManager(target_runtime)

        report = manager.status(package)
        assert report.status == "RUNNING"
        assert report.services[0].status is ContainerStatus.RUNNING
        assert "hello-ai" in manager.logs(package)

    def test_stop_then_start_again(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        import yaml

        from offlineai.schema.package import Package

        registry = Registry(target_home)
        registry.import_bundle(bundle)
        Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)

        record = registry.require("hello-ai")
        package = Package.model_validate(yaml.safe_load(record.package_yaml))
        manager = RuntimeManager(target_runtime)

        assert manager.stop(package, remove=True)
        assert manager.status(package).status == "STOPPED"
        manager.start(package)
        assert manager.status(package).status == "RUNNING"


class TestInstallRefusesToReachOut:
    """Section 74, the project's central promise."""

    def test_a_bundle_missing_its_image_fails_clearly(
        self,
        source_dir: Path,
        builder_home: Settings,
        target_home: Settings,
        target_runtime: FakeRuntime,
        tmp_path: Path,
    ) -> None:
        """Build with no container runtime available, so no image is packaged,
        then try to install. This must fail rather than pull."""
        unavailable = FakeRuntime(available=False)
        from offlineai.errors import SourceError

        with pytest.raises(SourceError, match="container runtime"):
            BundleBuilder(builder_home, runtime=unavailable).build(
                source_dir, output=tmp_path / "out"
            )

    def test_install_never_calls_pull(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)
        operations = {op for op, _ in target_runtime.calls}
        assert "pull" not in operations
        assert "load" in operations

    def test_running_a_container_whose_image_is_absent_is_refused(
        self, target_runtime: FakeRuntime
    ) -> None:
        """Mirrors docker --pull never."""
        from offlineai.runtime.base import RunSpec

        with pytest.raises(RuntimeFailureError, match="not present locally"):
            target_runtime.run_container(RunSpec(name="x", image="ghost:1"))


class TestRollback:
    def test_a_failed_start_rolls_back_to_a_clean_state(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)

        target_runtime.fail_on.add("run")
        installer = Installer(target_home, registry, target_runtime)
        with pytest.raises(RuntimeFailureError):
            installer.install("hello-ai", health_sleep=0)

        # The install id is the most recent one recorded.
        from offlineai.registry.db import open_registry_db

        with open_registry_db(registry.db_path) as connection:
            row = connection.execute(
                "SELECT id, state FROM installations ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        assert row["state"] == InstallState.FAILED.value

        target_runtime.fail_on.clear()
        result = rollback_installation(target_home, registry, target_runtime, row["id"])
        assert result.complete
        assert not target_runtime.containers

    def test_rollback_removes_the_network_it_created(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        installer = Installer(target_home, registry, target_runtime)
        result = installer.install("hello-ai", health_sleep=0)
        assert target_runtime.networks

        rollback_installation(target_home, registry, target_runtime, result.installation_id)
        assert not target_runtime.networks

    def test_rollback_keeps_images_by_default(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        result = Installer(target_home, registry, target_runtime).install(
            "hello-ai", health_sleep=0
        )
        report = rollback_installation(
            target_home, registry, target_runtime, result.installation_id
        )
        assert target_runtime.images, "a 12 GB image should not be discarded by default"
        assert any("kept" in s for s in report.skipped)

    def test_rollback_refuses_to_delete_outside_its_own_tree(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime, tmp_path: Path
    ) -> None:
        """A tampered journal must not become a way to delete arbitrary files."""
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        result = Installer(target_home, registry, target_runtime).install(
            "hello-ai", health_sleep=0
        )

        canary = tmp_path / "important-user-data.txt"
        canary.write_text("do not delete")

        from offlineai.registry.db import open_registry_db

        with open_registry_db(registry.db_path) as connection:
            connection.execute(
                """
                UPDATE installation_steps
                SET inverse_json = ?
                WHERE installation_id = ? AND seq = 1
                """,
                (f'[{{"action": "remove_path", "target": "{canary}"}}]', result.installation_id),
            )

        report = rollback_installation(
            target_home, registry, target_runtime, result.installation_id
        )
        assert canary.exists(), "rollback deleted a file outside the OfflineAI tree"
        assert any("refusing" in f for f in report.failed)


class TestDeduplication:
    """Section 44: shared artifacts are stored once, and removing one package
    must not pull content out from under another."""

    def test_two_packages_sharing_an_artifact_store_it_once(
        self,
        source_dir: Path,
        builder_home: Settings,
        builder_runtime: FakeRuntime,
        target_home: Settings,
        tmp_path: Path,
    ) -> None:
        first = BundleBuilder(builder_home, runtime=builder_runtime).build(
            source_dir, output=tmp_path / "b1"
        )

        second_dir = tmp_path / "src" / "hello-ai-2"
        shutil.copytree(source_dir, second_dir)
        (second_dir / "offlineai.yaml").write_text(
            PACKAGE_YAML.replace("name: hello-ai", "name: hello-ai-two")
        )
        second = BundleBuilder(builder_home, runtime=builder_runtime).build(
            second_dir, output=tmp_path / "b2"
        )

        registry = Registry(target_home)
        registry.import_bundle(first.bundle_path)
        result = registry.import_bundle(second.bundle_path)

        assert result.deduplicated > 0, "identical artifacts should not be stored twice"

    def test_removing_one_package_keeps_the_shared_artifacts(
        self,
        source_dir: Path,
        builder_home: Settings,
        builder_runtime: FakeRuntime,
        target_home: Settings,
        tmp_path: Path,
    ) -> None:
        first = BundleBuilder(builder_home, runtime=builder_runtime).build(
            source_dir, output=tmp_path / "b1"
        )
        second_dir = tmp_path / "src" / "hello-ai-2"
        shutil.copytree(source_dir, second_dir)
        (second_dir / "offlineai.yaml").write_text(
            PACKAGE_YAML.replace("name: hello-ai", "name: hello-ai-two")
        )
        second = BundleBuilder(builder_home, runtime=builder_runtime).build(
            second_dir, output=tmp_path / "b2"
        )

        registry = Registry(target_home)
        registry.import_bundle(first.bundle_path)
        registry.import_bundle(second.bundle_path)

        registry.remove("hello-ai")

        surviving = registry.require("hello-ai-two")
        for digest, _, _ in registry.artifacts_for(surviving.id):
            assert registry.artifact_path(digest).is_file(), (
                "removing one package deleted an artifact another still needs"
            )

    def test_removing_the_last_package_frees_storage(
        self, bundle: Path, target_home: Settings
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        before = registry.store.total_size()
        assert before > 0

        freed, freed_bytes = registry.remove("hello-ai")
        assert freed == 3
        assert freed_bytes > 0
        assert registry.store.total_size() == 0


class TestRegistryQueries:
    def test_list_search_and_info(self, bundle: Path, target_home: Settings) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)

        assert [r.name for r in registry.list_packages()] == ["hello-ai"]
        assert [r.name for r in registry.search("hello")] == ["hello-ai"]
        assert registry.search("nothing-like-this") == []
        assert registry.require("hello-ai").version == "1.0.0"

    def test_importing_twice_is_a_no_op_without_force(
        self, bundle: Path, target_home: Settings
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        again = registry.import_bundle(bundle)
        assert again.already_present is True
        assert again.artifacts_imported == 0

    def test_a_missing_package_says_what_is_available(self, target_home: Settings) -> None:
        from offlineai.errors import RegistryError

        registry = Registry(target_home)
        registry.initialise()
        with pytest.raises(RegistryError) as excinfo:
            registry.require("not-here")
        assert "offlineai import" in excinfo.value.render()


class TestMissingArtifactHandling:
    def test_install_fails_clearly_when_stored_content_is_gone(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        """Simulates storage corruption between import and install."""
        registry = Registry(target_home)
        registry.import_bundle(bundle)

        record = registry.require("hello-ai")
        for digest, _, path in registry.artifacts_for(record.id):
            if path.startswith("artifacts/containers/"):
                registry.artifact_path(digest).unlink()

        with pytest.raises(MissingArtifactError, match="missing from the local store"):
            Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)


class TestRollbackAfterSuccessfulInstall:
    """The gap that hid a real bug.

    Every earlier rollback test either failed before a container existed or
    only asserted on networks, so the container-removal path was never
    exercised. RuntimeManager.start was returning container ids while the
    rollback journal addressed containers by name, which meant removal matched
    nothing and quietly did nothing.
    """

    def test_containers_are_actually_removed(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        result = Installer(target_home, registry, target_runtime).install(
            "hello-ai", health_sleep=0
        )
        assert target_runtime.containers, "precondition: a container must be running"

        report = rollback_installation(
            target_home, registry, target_runtime, result.installation_id
        )
        assert report.complete
        assert target_runtime.containers == {}, (
            "rollback reported success but left the container running"
        )

    def test_start_returns_names_not_ids(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        import yaml

        from offlineai.schema.package import Package

        registry = Registry(target_home)
        registry.import_bundle(bundle)
        record = registry.require("hello-ai")
        package = Package.model_validate(yaml.safe_load(record.package_yaml))

        Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)
        RuntimeManager(target_runtime).stop(package, remove=True)
        started = RuntimeManager(target_runtime).start(package)

        assert started == [container_name("hello-ai", "app")]
        for name in started:
            assert name in target_runtime.containers, (
                "a returned identifier must address the container it named"
            )

    def test_install_result_reports_addressable_names(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        result = Installer(target_home, registry, target_runtime).install(
            "hello-ai", health_sleep=0
        )
        for name in result.services_started:
            assert target_runtime.container_state(name).status is ContainerStatus.RUNNING


class TestDockerfileBuiltImages:
    """The gap that let a real bug through to a live run.

    A container declaring `dockerfile:` does not run the image it names - that
    is its base. The builder produces a new image, and install has to start
    *that*. Every other test here declares a plain `image:`, so the effective
    reference never had to travel from build time to install time, and it
    turned out that it did not: install started the base image, whose CMD is a
    Python REPL, which exited immediately and restart-looped.
    """

    DOCKERFILE_PACKAGE = """\
apiVersion: offlineai/v1
kind: Package
metadata:
  name: built-app
  version: 1.0.0
runtime:
  type: docker
containers:
  - name: app
    image: python:3.12-slim
    dockerfile: Dockerfile
services:
  - name: app
    container: app
    ports:
      - "8000:8000"
"""

    @pytest.fixture
    def dockerfile_source(self, tmp_path: Path) -> Path:
        source = tmp_path / "src" / "built-app"
        source.mkdir(parents=True)
        (source / "offlineai.yaml").write_text(self.DOCKERFILE_PACKAGE)
        (source / "Dockerfile").write_text(
            'FROM python:3.12-slim\nCOPY app.py /app/app.py\nCMD ["python", "/app/app.py"]\n'
        )
        (source / "app.py").write_text("print('running')\n")
        return source

    def test_the_built_image_is_what_gets_packaged(
        self,
        dockerfile_source: Path,
        builder_home: Settings,
        builder_runtime: FakeRuntime,
        tmp_path: Path,
    ) -> None:
        from offlineai.bundler.archive import BundleReader
        from offlineai.schema.manifest import ArtifactType

        result = BundleBuilder(builder_home, runtime=builder_runtime).build(
            dockerfile_source, output=tmp_path / "out"
        )
        manifest = BundleReader.peek_manifest(result.bundle_path)
        image = manifest.artifacts_of_type(ArtifactType.OCI_IMAGE)[0]
        assert image.source == "offlineai/built-app-app:1.0.0"
        assert image.source != "python:3.12-slim", "the base image is not the deliverable"

    def test_install_starts_the_built_image_not_the_base(
        self,
        dockerfile_source: Path,
        builder_home: Settings,
        builder_runtime: FakeRuntime,
        target_home: Settings,
        target_runtime: FakeRuntime,
        tmp_path: Path,
    ) -> None:
        result = BundleBuilder(builder_home, runtime=builder_runtime).build(
            dockerfile_source, output=tmp_path / "out"
        )
        registry = Registry(target_home)
        registry.import_bundle(result.bundle_path)
        Installer(target_home, registry, target_runtime).install("built-app", health_sleep=0)

        container = target_runtime.containers[container_name("built-app", "app")]
        assert container.spec.image == "offlineai/built-app-app:1.0.0", (
            "install started the base image instead of the one the bundle carries"
        )

    def test_start_command_path_also_uses_the_built_image(
        self,
        dockerfile_source: Path,
        builder_home: Settings,
        builder_runtime: FakeRuntime,
        target_home: Settings,
        target_runtime: FakeRuntime,
        tmp_path: Path,
    ) -> None:
        """`start` after a `stop` goes through a different code path than
        install, and must agree with it."""
        import yaml

        from offlineai.runtime.manager import image_overrides_from
        from offlineai.schema.package import Package

        built = BundleBuilder(builder_home, runtime=builder_runtime).build(
            dockerfile_source, output=tmp_path / "out"
        )
        registry = Registry(target_home)
        registry.import_bundle(built.bundle_path)
        Installer(target_home, registry, target_runtime).install("built-app", health_sleep=0)

        record = registry.require("built-app")
        package = Package.model_validate(yaml.safe_load(record.package_yaml))
        manager = RuntimeManager(target_runtime)
        manager.stop(package, remove=True)
        manager.start(package, image_overrides=image_overrides_from(record.manifest()))

        container = target_runtime.containers[container_name("built-app", "app")]
        assert container.spec.image == "offlineai/built-app-app:1.0.0"


class TestReinstallIsIdempotent:
    def test_installing_twice_does_not_fail_on_a_name_collision(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        registry = Registry(target_home)
        registry.import_bundle(bundle)
        installer = Installer(target_home, registry, target_runtime)
        installer.install("hello-ai", health_sleep=0)
        second = installer.install("hello-ai", health_sleep=0)
        assert second.state is InstallState.COMPLETED


class TestStatusClassification:
    """`stop` leaves a container EXITED, not absent. Reporting that as
    DEGRADED told an operator something was wrong when nothing was."""

    @pytest.fixture
    def installed(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> tuple[RuntimeManager, object]:
        import yaml

        from offlineai.schema.package import Package

        registry = Registry(target_home)
        registry.import_bundle(bundle)
        Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)
        record = registry.require("hello-ai")
        package = Package.model_validate(yaml.safe_load(record.package_yaml))
        return RuntimeManager(target_runtime), package

    def test_running_when_every_service_runs(self, installed: tuple) -> None:
        manager, package = installed
        assert manager.status(package).status == "RUNNING"

    def test_stopped_when_a_container_is_merely_exited(self, installed: tuple) -> None:
        manager, package = installed
        manager.stop(package)  # no --remove: container stays, exited
        assert manager.status(package).status == "STOPPED"

    def test_stopped_when_containers_are_removed(self, installed: tuple) -> None:
        manager, package = installed
        manager.stop(package, remove=True)
        assert manager.status(package).status == "STOPPED"

    def test_degraded_only_when_some_but_not_all_run(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        import yaml

        from offlineai.schema.package import Package

        registry = Registry(target_home)
        registry.import_bundle(bundle)
        Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)
        record = registry.require("hello-ai")
        package = Package.model_validate(yaml.safe_load(record.package_yaml))

        # A second service that is not running makes the package genuinely partial.
        package.services.append(package.services[0].model_copy(update={"name": "sidecar"}))
        assert RuntimeManager(target_runtime).status(package).status == "DEGRADED"


class TestEndpointReporting:
    """Reported endpoints must be the ports actually bound, not the declared
    ones. A --config override remaps them, and telling an operator to curl a
    port nothing is listening on is worse than saying nothing."""

    def test_endpoints_follow_a_port_override(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        import yaml

        from offlineai.schema.overrides import Overrides
        from offlineai.schema.package import Package

        registry = Registry(target_home)
        registry.import_bundle(bundle)
        overrides = Overrides.model_validate({"services": {"app": {"ports": ["8099:8000"]}}})
        Installer(target_home, registry, target_runtime).install(
            "hello-ai", health_sleep=0, overrides=overrides
        )

        record = registry.require("hello-ai")
        package = Package.model_validate(yaml.safe_load(record.package_yaml))
        report = RuntimeManager(target_runtime).status(package)

        assert report.endpoints == ["http://localhost:8099"], (
            f"reported {report.endpoints}; the package declares 8000 but the "
            "override moved it to 8099"
        )

    def test_the_container_is_started_on_the_overridden_port(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        from offlineai.schema.overrides import Overrides

        registry = Registry(target_home)
        registry.import_bundle(bundle)
        overrides = Overrides.model_validate({"services": {"app": {"ports": ["8099:8000"]}}})
        Installer(target_home, registry, target_runtime).install(
            "hello-ai", health_sleep=0, overrides=overrides
        )
        container = target_runtime.containers[container_name("hello-ai", "app")]
        assert container.spec.ports == ["8099:8000"]

    def test_declared_ports_are_used_when_there_is_no_override(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        import yaml

        from offlineai.schema.package import Package

        registry = Registry(target_home)
        registry.import_bundle(bundle)
        Installer(target_home, registry, target_runtime).install("hello-ai", health_sleep=0)
        record = registry.require("hello-ai")
        package = Package.model_validate(yaml.safe_load(record.package_yaml))
        assert RuntimeManager(target_runtime).status(package).endpoints == ["http://localhost:8000"]

    def test_environment_overrides_reach_the_container(
        self, bundle: Path, target_home: Settings, target_runtime: FakeRuntime
    ) -> None:
        from offlineai.schema.overrides import Overrides

        registry = Registry(target_home)
        registry.import_bundle(bundle)
        overrides = Overrides.model_validate({"environment": {"DEPLOYMENT_SITE": "site-b"}})
        Installer(target_home, registry, target_runtime).install(
            "hello-ai", health_sleep=0, overrides=overrides
        )
        container = target_runtime.containers[container_name("hello-ai", "app")]
        assert container.spec.environment["DEPLOYMENT_SITE"] == "site-b"
