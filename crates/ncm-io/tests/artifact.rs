use ncm_io::{
    Artifact, CHECKPOINT_REVISION, FileRecord, Manifest, TensorDtype, TensorRecord, UPSTREAM,
    sha256,
};
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize},
};
use std::{collections::BTreeMap, fs};

fn fixture() -> (tempfile::TempDir, Manifest) {
    let directory = tempfile::tempdir().unwrap();
    let mut manifest = Manifest {
        schema_version: 1,
        profile: "sudoku-easy-inspection-v1".into(),
        upstream_commit: UPSTREAM.into(),
        checkpoint_repository: "hazyresearch/ncm-checkpoints".into(),
        checkpoint_revision: CHECKPOINT_REVISION.into(),
        files: BTreeMap::new(),
        tensors: BTreeMap::new(),
    };
    let raw: Vec<_> = [1.0f32, -2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    for name in [
        "codec.safetensors",
        "generator.safetensors",
        "training.safetensors",
    ] {
        let view = TensorView::new(Dtype::F32, vec![2], &raw).unwrap();
        let bytes = serialize([("x", view)], None).unwrap();
        fs::write(directory.path().join(name), &bytes).unwrap();
        manifest.files.insert(
            name.into(),
            FileRecord {
                bytes: bytes.len() as u64,
                sha256: sha256(&bytes),
            },
        );
        manifest.tensors.insert(
            name.into(),
            vec![TensorRecord {
                name: "x".into(),
                shape: vec![2],
                dtype: TensorDtype::F32,
                bytes: raw.len(),
                sha256: sha256(&raw),
            }],
        );
    }
    for name in ["config.json", "tokenizer.json", "provenance.json"] {
        let bytes = b"{}";
        fs::write(directory.path().join(name), bytes).unwrap();
        manifest.files.insert(
            name.into(),
            FileRecord {
                bytes: 2,
                sha256: sha256(bytes),
            },
        );
    }
    save(directory.path(), &manifest);
    (directory, manifest)
}
fn save(path: &std::path::Path, manifest: &Manifest) {
    fs::write(
        path.join("manifest.json"),
        serde_json::to_vec_pretty(manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn valid_copied_load_and_undeclared_access() {
    let (dir, _) = fixture();
    let artifact = Artifact::load(dir.path()).unwrap();
    assert_eq!(artifact.manifest().tensors.len(), 3);
    assert_eq!(artifact.file("config.json").unwrap(), b"{}");
    assert!(artifact.file("../secret").is_err());
    fs::remove_file(dir.path().join("config.json")).unwrap();
    assert_eq!(artifact.file("config.json").unwrap(), b"{}"); // backing bytes remain owned
}

#[test]
fn inference_view_validates_then_releases_training_state() {
    let (dir, _) = fixture();
    let artifact = Artifact::load_inference(dir.path()).unwrap();
    assert!(artifact.file("training.safetensors").is_err());
    assert!(!artifact.file("codec.safetensors").unwrap().is_empty());
    assert_eq!(artifact.manifest().tensors.len(), 3);
    fs::write(dir.path().join("training.safetensors"), b"tampered").unwrap();
    assert!(Artifact::load_inference(dir.path()).is_err());
}

#[test]
fn rejects_corruption_truncation_and_extra_file_entries() {
    for data in [b"[]".as_slice(), b"".as_slice()] {
        let (dir, _) = fixture();
        fs::write(dir.path().join("config.json"), data).unwrap();
        assert!(Artifact::load(dir.path()).is_err());
    }
    let (dir, mut manifest) = fixture();
    manifest.files.insert(
        "../escape".into(),
        FileRecord {
            bytes: 0,
            sha256: sha256(b""),
        },
    );
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
}

#[test]
fn rejects_tensor_shape_dtype_hash_duplicates_and_budget() {
    for mutation in 0..6 {
        let (dir, mut manifest) = fixture();
        let entries = manifest.tensors.get_mut("codec.safetensors").unwrap();
        match mutation {
            0 => entries[0].shape = vec![usize::MAX, 2],
            1 => entries[0].dtype = TensorDtype::I64,
            2 => entries[0].sha256 = sha256(b"bad"),
            3 => entries.push(TensorRecord {
                name: "x".into(),
                shape: vec![2],
                dtype: TensorDtype::F32,
                bytes: 8,
                sha256: entries[0].sha256.clone(),
            }),
            4 => manifest.files.get_mut("config.json").unwrap().bytes = u64::MAX,
            5 => manifest.schema_version = 2,
            _ => unreachable!(),
        }
        save(dir.path(), &manifest);
        assert!(Artifact::load(dir.path()).is_err(), "mutation {mutation}");
    }
}

#[test]
fn rejects_malformed_payload_even_with_matching_file_hash() {
    for name in ["codec.safetensors", "config.json"] {
        let (dir, mut manifest) = fixture();
        let bytes = b"invalid";
        fs::write(dir.path().join(name), bytes).unwrap();
        manifest.files.insert(
            name.into(),
            FileRecord {
                bytes: bytes.len() as u64,
                sha256: sha256(bytes),
            },
        );
        save(dir.path(), &manifest);
        assert!(Artifact::load(dir.path()).is_err());
    }
}

#[test]
fn rejects_unknown_manifest_fields() {
    let (dir, manifest) = fixture();
    let mut value = serde_json::to_value(manifest).unwrap();
    value["surprise"] = true.into();
    fs::write(
        dir.path().join("manifest.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    assert!(Artifact::load(dir.path()).is_err());
}

#[test]
fn rejects_duplicate_manifest_keys() {
    let (dir, manifest) = fixture();
    let text = serde_json::to_string(&manifest).unwrap();
    let record = serde_json::to_string(&manifest.files["config.json"]).unwrap();
    let duplicate = text.replacen(
        "\"files\":{",
        &format!("\"files\":{{\"config.json\":{record},"),
        1,
    );
    let error = serde_json::from_str::<Manifest>(&duplicate).unwrap_err();
    assert!(error.to_string().contains("duplicate key"));
    drop(dir);
}

#[cfg(unix)]
#[test]
fn rejects_symlink_files_and_roots() {
    let (dir, _) = fixture();
    fs::rename(dir.path().join("config.json"), dir.path().join("real.json")).unwrap();
    std::os::unix::fs::symlink("real.json", dir.path().join("config.json")).unwrap();
    assert!(Artifact::load(dir.path()).is_err());
    let other = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(dir.path(), other.path().join("alias")).unwrap();
    let alias = other.path().join("alias");
    for root in [
        alias.clone(),
        alias.join("."),
        std::path::PathBuf::from(format!("{}/", alias.display())),
    ] {
        assert!(Artifact::load(root).is_err());
    }
}

#[test]
fn sudoku_phase3_profiles_are_explicit_and_other_profiles_are_rejected() {
    let (directory, mut manifest) = fixture();
    for profile in [
        "sudoku-easy-inspection-v1",
        "sudoku-medium-inspection-v1",
        "sudoku-hard-inspection-v1",
    ] {
        manifest.profile = profile.into();
        save(directory.path(), &manifest);
        assert_eq!(
            Artifact::load(directory.path()).unwrap().manifest().profile,
            profile
        );
    }
    manifest.profile = "sudoku-other-inspection-v1".into();
    save(directory.path(), &manifest);
    assert!(Artifact::load(directory.path()).is_err());
}

#[test]
fn tinystories_has_a_separate_exact_inventory_and_budget() {
    let (directory, mut manifest) = fixture();
    manifest.profile = "tinystories-inference-v1".into();
    let bundle = fs::read(directory.path().join("codec.safetensors")).unwrap();
    fs::write(directory.path().join("inference.safetensors"), &bundle).unwrap();
    manifest.files.insert(
        "inference.safetensors".into(),
        FileRecord {
            bytes: bundle.len() as u64,
            sha256: sha256(&bundle),
        },
    );
    let entries = serde_json::from_value(
        serde_json::to_value(&manifest.tensors["codec.safetensors"]).unwrap(),
    )
    .unwrap();
    manifest
        .tensors
        .insert("inference.safetensors".into(), entries);
    for (name, bytes) in [
        ("tokenizer-manifest.json", b"{}".as_slice()),
        ("tokenizer.LICENSE", b"license".as_slice()),
    ] {
        fs::write(directory.path().join(name), bytes).unwrap();
        manifest.files.insert(
            name.into(),
            FileRecord {
                bytes: bytes.len() as u64,
                sha256: sha256(bytes),
            },
        );
    }
    save(directory.path(), &manifest);
    assert_eq!(
        Artifact::load(directory.path())
            .unwrap()
            .manifest()
            .tensors
            .len(),
        4
    );

    // The profile cannot drop the inference tensor bundle or accept an
    // undeclared extra file, even when all remaining files are valid.
    manifest.files.remove("inference.safetensors");
    save(directory.path(), &manifest);
    assert!(Artifact::load(directory.path()).is_err());
    manifest.files.insert(
        "inference.safetensors".into(),
        FileRecord {
            bytes: bundle.len() as u64,
            sha256: sha256(&bundle),
        },
    );
    manifest
        .files
        .get_mut("generator.safetensors")
        .unwrap()
        .bytes = 384 * 1024 * 1024 + 1;
    save(directory.path(), &manifest);
    assert!(Artifact::load(directory.path()).is_err());
    manifest
        .files
        .get_mut("generator.safetensors")
        .unwrap()
        .bytes = 350 * 1024 * 1024;
    manifest.files.get_mut("codec.safetensors").unwrap().bytes = 350 * 1024 * 1024;
    save(directory.path(), &manifest);
    assert!(Artifact::load(directory.path()).is_err());

    // The higher TinyStories allowance never changes the Sudoku bound.
    let (sudoku_dir, mut sudoku) = fixture();
    sudoku.files.get_mut("generator.safetensors").unwrap().bytes = 256 * 1024 * 1024 + 1;
    save(sudoku_dir.path(), &sudoku);
    assert!(Artifact::load(sudoku_dir.path()).is_err());
}
