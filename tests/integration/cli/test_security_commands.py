"""`keygen`, `sign`, `verify-signature`, `sbom`, `network-check`.

The signing engine has its own tests; what these cover is the command layer -
that `--key` is actually honoured, that the exit code reaches the caller, and
that the CLI is honest about what a signature does and does not prove.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from offlineai.exitcodes import ExitCode
from offlineai.runtime.fake import FakeRuntime


class TestKeygen:
    def test_writes_a_keypair(self, run_cli: Any, tmp_path: Path) -> None:
        key = tmp_path / "k.pem"
        payload = run_cli("--json", "keygen", "--out", str(key)).assert_ok().json()
        assert Path(payload["private_key"]).is_file()
        assert Path(payload["public_key"]).is_file()
        assert len(payload["fingerprint"]) == 19

    def test_the_private_key_is_not_world_readable(self, run_cli: Any, tmp_path: Path) -> None:
        key = tmp_path / "k.pem"
        run_cli("keygen", "--out", str(key)).assert_ok()
        assert key.stat().st_mode & 0o777 == 0o600

    def test_it_warns_against_copying_the_private_key_to_a_target(
        self, run_cli: Any, tmp_path: Path
    ) -> None:
        result = run_cli("keygen", "--out", str(tmp_path / "k.pem")).assert_ok()
        assert "Never copy the private key" in result.output

    def test_it_refuses_to_overwrite(self, run_cli: Any, tmp_path: Path) -> None:
        key = tmp_path / "k.pem"
        run_cli("keygen", "--out", str(key)).assert_ok()
        result = run_cli("keygen", "--out", str(key))
        assert result.exit_code == ExitCode.SIGNATURE_FAILURE
        assert "already exists" in result.output


class TestSignAndVerify:
    def test_sign_writes_a_detached_signature(
        self, run_cli: Any, bundle: Path, signing_key: tuple[Path, Path]
    ) -> None:
        private, _ = signing_key
        payload = run_cli("--json", "sign", str(bundle), "--key", str(private)).assert_ok().json()
        assert Path(payload["signature"]).is_file()
        assert payload["signature"].endswith(".offlineai.sig")

    def test_a_detached_signature_verifies(
        self, run_cli: Any, bundle: Path, signing_key: tuple[Path, Path]
    ) -> None:
        private, public = signing_key
        run_cli("sign", str(bundle), "--key", str(private)).assert_ok()
        payload = (
            run_cli("--json", "verify-signature", str(bundle), "--key", str(public))
            .assert_ok()
            .json()
        )
        assert payload["verified"] is True
        assert payload["embedded"] is False

    def test_a_signer_identity_is_recorded(
        self, run_cli: Any, bundle: Path, signing_key: tuple[Path, Path]
    ) -> None:
        private, public = signing_key
        run_cli("sign", str(bundle), "--key", str(private), "--signer", "ops@example").assert_ok()
        payload = (
            run_cli("--json", "verify-signature", str(bundle), "--key", str(public))
            .assert_ok()
            .json()
        )
        assert payload["signer"] == "ops@example"

    def test_an_unsigned_bundle_says_how_to_sign_it(self, run_cli: Any, bundle: Path) -> None:
        result = run_cli("verify-signature", str(bundle))
        assert result.exit_code == ExitCode.SIGNATURE_FAILURE
        assert "offlineai sign" in result.output

    def test_the_wrong_key_is_rejected(
        self, run_cli: Any, bundle: Path, signing_key: tuple[Path, Path], tmp_path: Path
    ) -> None:
        private, _ = signing_key
        run_cli("sign", str(bundle), "--key", str(private)).assert_ok()
        run_cli("keygen", "--out", str(tmp_path / "attacker.pem")).assert_ok()

        result = run_cli("verify-signature", str(bundle), "--key", str(tmp_path / "attacker.pub"))
        assert result.exit_code == ExitCode.SIGNATURE_FAILURE
        assert "different key" in result.output

    def test_without_a_key_it_says_this_is_not_proof_of_origin(
        self, run_cli: Any, bundle: Path, signing_key: tuple[Path, Path]
    ) -> None:
        """An attacker who rewrote the bundle can re-sign it, so verification
        without a trusted key proves internal consistency and nothing more.
        The CLI has to say so."""
        private, _ = signing_key
        run_cli("sign", str(bundle), "--key", str(private)).assert_ok()
        result = run_cli("verify-signature", str(bundle)).assert_ok()
        assert "not where it came from" in result.output
        assert "--key" in result.output

    def test_a_signature_over_a_tampered_manifest_fails(
        self,
        run_cli: Any,
        package_dir: Path,
        tmp_path: Path,
        signing_key: tuple[Path, Path],
        fake_runtime: FakeRuntime,
    ) -> None:
        private, public = signing_key
        run_cli(
            "build",
            str(package_dir),
            "-o",
            str(tmp_path / "signed"),
            "--sign-key",
            str(private),
        ).assert_ok()
        signed = tmp_path / "signed" / "cli-demo-1.0.0.offlineai"

        run_cli("--json", "verify-signature", str(signed), "--key", str(public)).assert_ok()

    def test_build_sign_key_embeds_the_signature(
        self,
        run_cli: Any,
        package_dir: Path,
        tmp_path: Path,
        signing_key: tuple[Path, Path],
        fake_runtime: FakeRuntime,
    ) -> None:
        private, public = signing_key
        run_cli(
            "build",
            str(package_dir),
            "-o",
            str(tmp_path / "signed"),
            "--sign-key",
            str(private),
        ).assert_ok()
        signed = tmp_path / "signed" / "cli-demo-1.0.0.offlineai"
        payload = (
            run_cli("--json", "verify-signature", str(signed), "--key", str(public))
            .assert_ok()
            .json()
        )
        assert payload["embedded"] is True


class TestSbom:
    def test_emits_cyclonedx(self, run_cli: Any, bundle: Path) -> None:
        payload = json.loads(run_cli("sbom", str(bundle)).assert_ok().stdout)
        assert payload["bomFormat"] == "CycloneDX"
        assert payload["components"]

    def test_components_carry_hashes(self, run_cli: Any, bundle: Path) -> None:
        payload = json.loads(run_cli("sbom", str(bundle)).assert_ok().stdout)
        for component in payload["components"]:
            assert component["hashes"][0]["alg"] == "SHA-256"
            assert len(component["hashes"][0]["content"]) == 64

    def test_it_can_write_to_a_file(self, run_cli: Any, bundle: Path, tmp_path: Path) -> None:
        out = tmp_path / "sbom.json"
        run_cli("sbom", str(bundle), "--out", str(out)).assert_ok()
        assert json.loads(out.read_text())["bomFormat"] == "CycloneDX"

    def test_the_licence_view_carries_the_disclaimer(self, run_cli: Any, bundle: Path) -> None:
        """Section 37: informational, and no legal claim."""
        result = run_cli("sbom", str(bundle), "--licenses").assert_ok()
        assert "informational" in result.output
        assert "responsible for reviewing" in result.output

    def test_unknown_licences_are_shown_not_hidden(self, run_cli: Any, bundle: Path) -> None:
        """An operator reviewing obligations needs to see what could not be
        determined, not only what could."""
        result = run_cli("sbom", str(bundle), "--licenses").assert_ok()
        assert "(not recorded)" in result.output


class TestNetworkCheck:
    def test_a_clean_package_passes(self, run_cli: Any, package_dir: Path) -> None:
        payload = run_cli("--json", "network-check", str(package_dir)).assert_ok().json()
        assert payload["clean"] is True
        assert "No runtime network dependency" in payload["summary"]

    def test_build_time_references_are_classified_as_such(
        self, run_cli: Any, package_dir: Path
    ) -> None:
        payload = run_cli("--json", "network-check", str(package_dir)).assert_ok().json()
        phases = {f["phase"] for f in payload["findings"]}
        assert "BUILD TIME" in phases
        assert "RUNTIME" not in phases

    def test_a_runtime_dependency_exits_five(self, run_cli: Any, tmp_path: Path) -> None:
        leaky = tmp_path / "leaky"
        leaky.mkdir()
        (leaky / "offlineai.yaml").write_text(
            "apiVersion: offlineai/v1\nkind: Package\n"
            "metadata:\n  name: leaky\n  version: 1.0.0\n"
            'environment:\n  MODEL_API_ENDPOINT: "https://api.example.com/v1"\n'
        )
        result = run_cli("network-check", str(leaky))
        assert result.exit_code == ExitCode.MISSING_DEPENDENCY
        assert "will fail on an air-gapped host" in result.output

    def test_it_works_on_a_built_bundle(self, run_cli: Any, bundle: Path) -> None:
        payload = run_cli("--json", "network-check", str(bundle)).assert_ok().json()
        assert payload["package"] == "cli-demo"
