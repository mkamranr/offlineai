"""Section 39: never blindly extract archive paths.

A bundle arrives from outside the security boundary - that is the entire
premise of the product - so extraction is treated as parsing hostile input.
These tests construct the malicious archives directly rather than shipping
binary fixtures, so the attack being defended against is readable.
"""

from __future__ import annotations

import io
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest

from offlineai.errors import UnsafeArchiveError
from offlineai.security.extraction import (
    ExtractionLimits,
    safe_extract,
    validate_member_name,
)


def _tar_with(build: Callable[[tarfile.TarFile], None]) -> tarfile.TarFile:
    """Build an in-memory tar and reopen it for reading."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        build(archive)
    buffer.seek(0)
    return tarfile.open(fileobj=buffer, mode="r")


def _add_file(archive: tarfile.TarFile, name: str, data: bytes = b"payload") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _add_link(archive: tarfile.TarFile, name: str, target: str, *, hard: bool = False) -> None:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE if hard else tarfile.SYMTYPE
    info.linkname = target
    archive.addfile(info)


class TestPathTraversal:
    @pytest.mark.parametrize(
        "name",
        [
            "../../etc/passwd",
            "../outside.txt",
            "a/../../outside.txt",
            "a/b/../../../outside.txt",
            "./../../outside.txt",
        ],
    )
    def test_parent_traversal_is_refused(self, tmp_path: Path, name: str) -> None:
        archive = _tar_with(lambda t: _add_file(t, name))
        with pytest.raises(UnsafeArchiveError):
            safe_extract(archive, tmp_path / "dest")

    @pytest.mark.parametrize("name", ["/etc/passwd", "//etc/passwd", "/tmp/evil"])
    def test_absolute_paths_are_refused(self, tmp_path: Path, name: str) -> None:
        archive = _tar_with(lambda t: _add_file(t, name))
        with pytest.raises(UnsafeArchiveError):
            safe_extract(archive, tmp_path / "dest")

    def test_nothing_is_written_outside_the_destination(self, tmp_path: Path) -> None:
        """The decisive assertion: even a refused extraction must not touch
        anything above the destination directory."""
        dest = tmp_path / "dest"
        canary = tmp_path / "outside.txt"
        archive = _tar_with(lambda t: _add_file(t, "../outside.txt", b"OWNED"))
        with pytest.raises(UnsafeArchiveError):
            safe_extract(archive, dest)
        assert not canary.exists()

    def test_validation_happens_before_any_write(self, tmp_path: Path) -> None:
        """A hostile member late in the archive must invalidate the whole
        extraction, not leave the earlier members on disk."""
        dest = tmp_path / "dest"

        def build(archive: tarfile.TarFile) -> None:
            _add_file(archive, "good.txt", b"fine")
            _add_file(archive, "../escape.txt", b"OWNED")

        with pytest.raises(UnsafeArchiveError):
            safe_extract(_tar_with(build), dest)
        assert not (dest / "good.txt").exists(), "partial extraction left files behind"


class TestLinks:
    def test_symlinks_are_refused_by_default(self, tmp_path: Path) -> None:
        archive = _tar_with(lambda t: _add_link(t, "link", "/etc/passwd"))
        with pytest.raises(UnsafeArchiveError, match="symlink"):
            safe_extract(archive, tmp_path / "dest")

    def test_relative_escaping_symlinks_are_refused(self, tmp_path: Path) -> None:
        archive = _tar_with(lambda t: _add_link(t, "link", "../../etc/passwd"))
        with pytest.raises(UnsafeArchiveError):
            safe_extract(archive, tmp_path / "dest")

    def test_hardlinks_are_refused(self, tmp_path: Path) -> None:
        archive = _tar_with(lambda t: _add_link(t, "link", "../../etc/passwd", hard=True))
        with pytest.raises(UnsafeArchiveError, match="hard link"):
            safe_extract(archive, tmp_path / "dest")

    def test_symlink_then_write_through_it_is_refused(self, tmp_path: Path) -> None:
        """The classic two-step: plant a directory symlink pointing outside,
        then write a file 'inside' it."""

        def build(archive: tarfile.TarFile) -> None:
            _add_link(archive, "sneaky", "..")
            _add_file(archive, "sneaky/owned.txt", b"OWNED")

        with pytest.raises(UnsafeArchiveError):
            safe_extract(_tar_with(build), tmp_path / "dest")
        assert not (tmp_path / "owned.txt").exists()

    def test_inward_symlinks_allowed_only_when_opted_in(self, tmp_path: Path) -> None:
        archive = _tar_with(lambda t: _add_link(t, "link", "real.txt"))
        limits = ExtractionLimits(allow_symlinks=True)
        safe_extract(archive, tmp_path / "dest", limits=limits)
        assert (tmp_path / "dest" / "link").is_symlink()


class TestSpecialFileTypes:
    @pytest.mark.parametrize(
        ("typeflag", "label"),
        [
            (tarfile.CHRTYPE, "character device"),
            (tarfile.BLKTYPE, "block device"),
            (tarfile.FIFOTYPE, "FIFO"),
        ],
    )
    def test_device_and_fifo_members_are_refused(
        self, tmp_path: Path, typeflag: bytes, label: str
    ) -> None:
        def build(archive: tarfile.TarFile) -> None:
            info = tarfile.TarInfo("node")
            info.type = typeflag
            archive.addfile(info)

        with pytest.raises(UnsafeArchiveError):
            safe_extract(_tar_with(build), tmp_path / "dest")


class TestResourceLimits:
    def test_member_count_is_capped(self, tmp_path: Path) -> None:
        def build(archive: tarfile.TarFile) -> None:
            for i in range(50):
                _add_file(archive, f"f{i}.txt", b"x")

        limits = ExtractionLimits(max_members=10)
        with pytest.raises(UnsafeArchiveError, match="member"):
            safe_extract(_tar_with(build), tmp_path / "dest", limits=limits)

    def test_total_size_is_capped(self, tmp_path: Path) -> None:
        def build(archive: tarfile.TarFile) -> None:
            for i in range(5):
                _add_file(archive, f"f{i}.bin", b"x" * 1000)

        limits = ExtractionLimits(max_total_bytes=2000)
        with pytest.raises(UnsafeArchiveError, match="size"):
            safe_extract(_tar_with(build), tmp_path / "dest", limits=limits)

    def test_declared_member_size_is_capped(self, tmp_path: Path) -> None:
        archive = _tar_with(lambda t: _add_file(t, "big.bin", b"x" * 5000))
        limits = ExtractionLimits(max_member_bytes=100)
        with pytest.raises(UnsafeArchiveError):
            safe_extract(archive, tmp_path / "dest", limits=limits)


class TestNameValidation:
    @pytest.mark.parametrize(
        "name",
        ["", ".", "..", "/", "a\x00b", "C:\\evil", "\\\\server\\share"],
    )
    def test_hostile_names_are_refused(self, name: str) -> None:
        with pytest.raises(UnsafeArchiveError):
            validate_member_name(name)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("manifest.yaml", "manifest.yaml"),
            ("./manifest.yaml", "manifest.yaml"),
            ("artifacts/models/qwen3/config.json", "artifacts/models/qwen3/config.json"),
            ("a//b", "a/b"),
            ("a/./b", "a/b"),
        ],
    )
    def test_ordinary_names_are_normalised(self, name: str, expected: str) -> None:
        assert validate_member_name(name) == expected


class TestHappyPath:
    def test_a_well_formed_archive_extracts(self, tmp_path: Path) -> None:
        def build(archive: tarfile.TarFile) -> None:
            _add_file(archive, "manifest.yaml", b"formatVersion: '1'\n")
            _add_file(archive, "artifacts/models/qwen3/config.json", b"{}")

        dest = tmp_path / "dest"
        safe_extract(_tar_with(build), dest)
        assert (dest / "manifest.yaml").read_bytes() == b"formatVersion: '1'\n"
        assert (dest / "artifacts/models/qwen3/config.json").read_bytes() == b"{}"

    def test_extracted_files_are_not_executable_by_default(self, tmp_path: Path) -> None:
        """A bundle should not be able to smuggle in a setuid or world-writable
        file through archive permission bits."""

        def build(archive: tarfile.TarFile) -> None:
            info = tarfile.TarInfo("script.sh")
            info.size = 2
            info.mode = 0o4777  # setuid + world-writable
            archive.addfile(info, io.BytesIO(b"hi"))

        dest = tmp_path / "dest"
        safe_extract(_tar_with(build), dest)
        mode = (dest / "script.sh").stat().st_mode & 0o7777
        assert not mode & 0o4000, "setuid bit survived extraction"
        assert not mode & 0o0002, "world-writable bit survived extraction"
