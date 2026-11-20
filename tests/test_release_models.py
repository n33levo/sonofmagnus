import io
import tarfile

import pytest

from model.artifacts import artifact_info
from scripts import fetch_models


def test_model_archive_is_verified_and_never_replaces_other_weights(tmp_path,monkeypatch):
    model=tmp_path/'source/model.mlx';model.mkdir(parents=True);(model/'weights').write_bytes(b'fixture')
    archive=tmp_path/'model.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(model,arcname='model.mlx')
    monkeypatch.setattr(fetch_models,'fetch',lambda *args:archive)
    entry=dict(file='model.tar.gz',directory='model.mlx',artifact_sha256=artifact_info(model)['sha256'])
    output=tmp_path/'ckpts';path=fetch_models.fetch_model(entry,output)
    assert artifact_info(path)==artifact_info(model)
    assert fetch_models.fetch_model(entry,output)==path
    (path/'weights').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Existing model differs'):fetch_models.fetch_model(entry,output)


def test_archive_cannot_escape_its_temporary_directory(tmp_path,monkeypatch):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        entry=tarfile.TarInfo('../../escape');entry.size=1;tar.addfile(entry,io.BytesIO(b'x'))
    monkeypatch.setattr(fetch_models,'fetch',lambda *args:archive)
    with pytest.raises(tarfile.FilterError):fetch_models.fetch_model(dict(file='bad.tar.gz',directory='model',artifact_sha256='unused'),tmp_path/'ckpts')
    assert not (tmp_path/'escape').exists()
