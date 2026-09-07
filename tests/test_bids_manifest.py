from pathlib import Path

from hybrid_cvr.io.bids import scan_bids_like, write_manifest


def test_bids_manifest_creation(tmp_path: Path):
    root = tmp_path / "data" / "raw"
    anat = root / "sub-01" / "ses-01" / "anat"
    cvr = root / "sub-01" / "ses-01" / "cvr"
    anat.mkdir(parents=True)
    cvr.mkdir(parents=True)
    (anat / "sub-01_ses-01_T1w.nii").write_text("fake")
    (cvr / "sub-01_ses-01_bold.nii.gz").write_text("fake")
    (cvr / "sub-01_ses-01_gas_traces.txt").write_text("fake")

    records = scan_bids_like(root, tmp_path / "processed")
    assert len(records) == 1
    assert records[0].subject == "sub-01"
    out = write_manifest(records, tmp_path / "manifest.csv")
    assert "subject,session,t1w_path,bold_path,gas_path,output_dir" in out.read_text()


def test_bids_manifest_reuses_subject_t1_for_repeat_session(tmp_path: Path):
    root = tmp_path / "data" / "raw"
    anat = root / "sub-01" / "ses-01" / "anat"
    cvr_1 = root / "sub-01" / "ses-01" / "cvr"
    cvr_2 = root / "sub-01" / "ses-02" / "cvr"
    anat.mkdir(parents=True)
    cvr_1.mkdir(parents=True)
    cvr_2.mkdir(parents=True)
    t1 = anat / "sub-01_ses-01_T1w.nii"
    t1.write_text("fake")
    (cvr_1 / "sub-01_ses-01_bold.nii.gz").write_text("fake")
    (cvr_1 / "sub-01_ses-01_gas_traces.txt").write_text("fake")
    (cvr_2 / "sub-01_ses-02_bold.nii.gz").write_text("fake")
    (cvr_2 / "sub-01_ses-02_gas_traces.txt").write_text("fake")

    records = scan_bids_like(root, tmp_path / "processed")
    assert len(records) == 2
    assert {record.session for record in records} == {"ses-01", "ses-02"}
    assert all(record.t1w_path == str(t1) for record in records)
