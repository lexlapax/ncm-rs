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
fn released_payload_cannot_be_accessed_or_released_twice() {
    let (dir, _) = fixture();
    let mut artifact = Artifact::load_inference(dir.path()).unwrap();
    artifact.release_file("codec.safetensors").unwrap();
    assert!(artifact.file("codec.safetensors").is_err());
    assert!(artifact.release_file("codec.safetensors").is_err());
    assert!(!artifact.file("generator.safetensors").unwrap().is_empty());
    assert_eq!(artifact.manifest().tensors.len(), 3);
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

    // Matching the outer file hash does not bypass the recorded tensor hash.
    let (dir, mut manifest) = fixture();
    let path = dir.path().join("training.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    *bytes.last_mut().unwrap() ^= 1;
    fs::write(&path, &bytes).unwrap();
    manifest
        .files
        .get_mut("training.safetensors")
        .unwrap()
        .sha256 = sha256(&bytes);
    save(dir.path(), &manifest);
    assert!(Artifact::load_inference(dir.path()).is_err());
    assert!(Artifact::load(dir.path()).is_err());
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
    for name in ["codec.safetensors", "training.safetensors", "config.json"] {
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
        assert!(Artifact::load_inference(dir.path()).is_err());
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

#[test]
fn xsum_profile_accepts_only_its_explicit_bundles_and_budget() {
    let (dir, mut manifest) = fixture();
    manifest.profile = "xsum-inference-v1".into();
    let training = fs::read(dir.path().join("training.safetensors")).unwrap();
    fs::write(dir.path().join("inference.safetensors"), &training).unwrap();
    manifest.files.insert(
        "inference.safetensors".into(),
        FileRecord {
            bytes: training.len() as u64,
            sha256: sha256(&training),
        },
    );
    let original = &manifest.tensors["training.safetensors"][0];
    manifest.tensors.insert(
        "inference.safetensors".into(),
        vec![TensorRecord {
            name: original.name.clone(),
            shape: original.shape.clone(),
            dtype: original.dtype,
            bytes: original.bytes,
            sha256: original.sha256.clone(),
        }],
    );
    fs::write(dir.path().join("tokenizer-manifest.json"), b"{}").unwrap();
    manifest.files.insert(
        "tokenizer-manifest.json".into(),
        FileRecord {
            bytes: 2,
            sha256: sha256(b"{}"),
        },
    );
    fs::write(dir.path().join("tokenizer.LICENSE"), b"notice").unwrap();
    manifest.files.insert(
        "tokenizer.LICENSE".into(),
        FileRecord {
            bytes: 6,
            sha256: sha256(b"notice"),
        },
    );
    save(dir.path(), &manifest);
    let full = Artifact::load(dir.path()).unwrap();
    let inference = Artifact::load_inference(dir.path()).unwrap();
    assert_eq!(full.manifest().profile, "xsum-inference-v1");
    assert!(inference.file("training.safetensors").is_err());
    assert!(!inference.file("inference.safetensors").unwrap().is_empty());
    manifest.profile = "xsum-inference-v2".into();
    save(dir.path(), &manifest);
    assert_eq!(
        Artifact::load_inference(dir.path())
            .unwrap()
            .manifest()
            .profile,
        "xsum-inference-v2"
    );
    manifest.files.get_mut("config.json").unwrap().bytes = 384 * 1024 * 1024 + 1;
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
    manifest.files.get_mut("config.json").unwrap().bytes = 2;
    manifest.files.remove("inference.safetensors");
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
}

#[test]
fn wmt14_profile_has_exact_inventory_and_independent_bounds() {
    let (dir, mut manifest) = fixture();
    manifest.profile = "wmt14-inference-v1".into();
    for (name, bytes) in [
        ("tokenizer-manifest.json", b"{}".as_slice()),
        ("tokenizer.LICENSE", b"notice".as_slice()),
    ] {
        fs::write(dir.path().join(name), bytes).unwrap();
        manifest.files.insert(
            name.into(),
            FileRecord {
                bytes: bytes.len() as u64,
                sha256: sha256(bytes),
            },
        );
    }
    save(dir.path(), &manifest);
    let full = Artifact::load(dir.path()).unwrap();
    let inference = Artifact::load_inference(dir.path()).unwrap();
    assert_eq!(full.manifest().profile, "wmt14-inference-v1");
    assert!(!full.file("training.safetensors").unwrap().is_empty());
    assert!(inference.file("training.safetensors").is_err());
    assert!(inference.file("inference.safetensors").is_err());

    let reusable = Artifact::load_wmt14_inference_reusable(dir.path()).unwrap();
    assert!(reusable.file("codec.safetensors").is_err());
    assert!(reusable.file("training.safetensors").is_err());
    let generator = inference.file("generator.safetensors").unwrap();
    let original = reusable.file("generator.safetensors").unwrap();
    assert_eq!(original, generator);
    let backing = original.as_ptr();
    let reusable = reusable.with_wmt14_bundle("codec.safetensors").unwrap();
    assert!(reusable.file("generator.safetensors").is_err());
    let codec = reusable.file("codec.safetensors").unwrap();
    assert_eq!(codec, inference.file("codec.safetensors").unwrap());
    assert_eq!(codec.as_ptr(), backing);
    let reusable = reusable.with_wmt14_bundle("generator.safetensors").unwrap();
    assert_eq!(
        reusable.file("generator.safetensors").unwrap().as_ptr(),
        backing
    );
    let reusable = reusable.with_wmt14_bundle("codec.safetensors").unwrap();
    let mut corrupt = generator.to_vec();
    *corrupt.last_mut().unwrap() ^= 1;
    let generator_path = dir.path().join("generator.safetensors");
    fs::write(&generator_path, corrupt).unwrap();
    assert!(reusable.with_wmt14_bundle("generator.safetensors").is_err());
    fs::write(generator_path, generator).unwrap();

    manifest.files.insert(
        "inference.safetensors".into(),
        FileRecord {
            bytes: 0,
            sha256: sha256(b""),
        },
    );
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
    manifest.files.remove("inference.safetensors");
    manifest.files.get_mut("codec.safetensors").unwrap().bytes = 640 * 1024 * 1024 + 1;
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
    manifest.files.get_mut("codec.safetensors").unwrap().bytes = 639 * 1024 * 1024;
    manifest
        .files
        .get_mut("generator.safetensors")
        .unwrap()
        .bytes = 639 * 1024 * 1024;
    manifest
        .files
        .get_mut("training.safetensors")
        .unwrap()
        .bytes = 3 * 1024 * 1024;
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
}

/// A valid LM1B-small inventory built from the shared fixture, saved to disk.
fn lm1b_small_fixture() -> (tempfile::TempDir, Manifest) {
    let (dir, mut manifest) = fixture();
    manifest.profile = "lm1b-small-inference-v1".into();
    let training = fs::read(dir.path().join("training.safetensors")).unwrap();
    fs::write(dir.path().join("inference.safetensors"), &training).unwrap();
    manifest.files.insert(
        "inference.safetensors".into(),
        FileRecord {
            bytes: training.len() as u64,
            sha256: sha256(&training),
        },
    );
    let original = &manifest.tensors["training.safetensors"][0];
    manifest.tensors.insert(
        "inference.safetensors".into(),
        vec![TensorRecord {
            name: original.name.clone(),
            shape: original.shape.clone(),
            dtype: original.dtype,
            bytes: original.bytes,
            sha256: original.sha256.clone(),
        }],
    );
    for (name, bytes) in [
        ("tokenizer-manifest.json", b"{}".as_slice()),
        ("tokenizer.LICENSE", b"notice".as_slice()),
    ] {
        fs::write(dir.path().join(name), bytes).unwrap();
        manifest.files.insert(
            name.into(),
            FileRecord {
                bytes: bytes.len() as u64,
                sha256: sha256(bytes),
            },
        );
    }
    save(dir.path(), &manifest);
    (dir, manifest)
}

#[test]
fn lm1b_small_profile_has_independent_inventory_and_bounds() {
    let (dir, mut manifest) = lm1b_small_fixture();
    let training = fs::read(dir.path().join("training.safetensors")).unwrap();
    save(dir.path(), &manifest);
    let full = Artifact::load(dir.path()).unwrap();
    let inference = Artifact::load_inference(dir.path()).unwrap();
    assert_eq!(full.manifest().profile, "lm1b-small-inference-v1");
    assert!(!full.file("training.safetensors").unwrap().is_empty());
    assert!(inference.file("training.safetensors").is_err());
    assert!(!inference.file("inference.safetensors").unwrap().is_empty());
    manifest.files.remove("inference.safetensors");
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
    manifest.files.insert(
        "inference.safetensors".into(),
        FileRecord {
            bytes: training.len() as u64,
            sha256: sha256(&training),
        },
    );
    manifest.files.get_mut("codec.safetensors").unwrap().bytes = 448 * 1024 * 1024 + 1;
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
    for name in [
        "codec.safetensors",
        "generator.safetensors",
        "training.safetensors",
    ] {
        manifest.files.get_mut(name).unwrap().bytes = 400 * 1024 * 1024;
    }
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
    manifest.profile = "lm1b-medium-inference-v1".into();
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
}

#[test]
fn lm1b_sizes_have_separate_budgets() {
    // The same small payload is a valid medium or large inventory; each size keeps its own limits.
    for (profile, max_file, max_total) in [
        ("lm1b-small-inference-v1", 448_u64, 1024_u64),
        ("lm1b-medium-inference-v1", 1280, 1920),
        ("lm1b-large-inference-v1", 2944, 3584),
    ] {
        let (dir, mut manifest) = lm1b_small_fixture();
        manifest.profile = profile.into();
        save(dir.path(), &manifest);
        assert_eq!(
            Artifact::load_inference(dir.path())
                .unwrap()
                .manifest()
                .profile,
            profile
        );
        let original = manifest.files["generator.safetensors"].bytes;
        manifest
            .files
            .get_mut("generator.safetensors")
            .unwrap()
            .bytes = max_file * 1024 * 1024 + 1;
        save(dir.path(), &manifest);
        assert!(
            Artifact::load(dir.path()).is_err(),
            "{profile} per-file limit"
        );
        let share = max_total * 1024 * 1024 / 3 + 1;
        for name in [
            "codec.safetensors",
            "generator.safetensors",
            "training.safetensors",
        ] {
            manifest.files.get_mut(name).unwrap().bytes = share;
        }
        save(dir.path(), &manifest);
        assert!(Artifact::load(dir.path()).is_err(), "{profile} total limit");
        assert!(original < max_file * 1024 * 1024);
    }
    let (dir, mut manifest) = lm1b_small_fixture();
    manifest.profile = "lm1b-huge-inference-v1".into();
    save(dir.path(), &manifest);
    assert!(Artifact::load(dir.path()).is_err());
}

#[test]
fn owt_artifacts_are_rejected_until_a_checkpoint_is_released() {
    for profile in ["owt-inference-v1", "openwebtext-inference-v1"] {
        let (dir, mut manifest) = lm1b_small_fixture();
        manifest.profile = profile.into();
        save(dir.path(), &manifest);
        assert!(Artifact::load(dir.path()).is_err(), "{profile} accepted");
        assert!(
            Artifact::load_inference(dir.path()).is_err(),
            "{profile} accepted"
        );
    }
}

#[cfg(unix)]
#[test]
fn streaming_inference_matches_copied_load_across_small_and_parallel_bundles() {
    // Include tensors spanning both single-pass (< 4 MiB) and parallel (>= 4 MiB) paths,
    // plus an attn_mask_for_masking containing -inf which must be permitted.
    let (dir, mut manifest) = fixture();
    let big_a: Vec<u8> = (0..1_100_000usize)
        .flat_map(|i| ((i % 97) as f32 * 0.25).to_le_bytes())
        .collect();
    let big_b: Vec<u8> = (0..200_000usize)
        .flat_map(|i| (-((i % 53) as f32) * 0.5).to_le_bytes())
        .collect();
    let mask_raw: Vec<u8> = [0.0f32, f32::NEG_INFINITY, 0.0, 0.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let gen_bytes = serialize(
        [
            (
                "w_a",
                TensorView::new(Dtype::F32, vec![1_100_000], &big_a).unwrap(),
            ),
            (
                "w_b",
                TensorView::new(Dtype::F32, vec![200_000], &big_b).unwrap(),
            ),
            (
                "attn_mask_for_masking",
                TensorView::new(Dtype::F32, vec![1, 1, 2, 2], &mask_raw).unwrap(),
            ),
        ],
        None,
    )
    .unwrap();
    assert!(gen_bytes.len() > 4 * 1024 * 1024);
    fs::write(dir.path().join("generator.safetensors"), &gen_bytes).unwrap();
    manifest.files.insert(
        "generator.safetensors".into(),
        FileRecord {
            bytes: gen_bytes.len() as u64,
            sha256: sha256(&gen_bytes),
        },
    );
    manifest.tensors.insert(
        "generator.safetensors".into(),
        vec![
            TensorRecord {
                name: "w_a".into(),
                shape: vec![1_100_000],
                dtype: TensorDtype::F32,
                bytes: big_a.len(),
                sha256: sha256(&big_a),
            },
            TensorRecord {
                name: "w_b".into(),
                shape: vec![200_000],
                dtype: TensorDtype::F32,
                bytes: big_b.len(),
                sha256: sha256(&big_b),
            },
            TensorRecord {
                name: "attn_mask_for_masking".into(),
                shape: vec![1, 1, 2, 2],
                dtype: TensorDtype::F32,
                bytes: mask_raw.len(),
                sha256: sha256(&mask_raw),
            },
        ],
    );
    save(dir.path(), &manifest);

    let copied = Artifact::load_inference(dir.path()).unwrap();
    let mut streamed = Artifact::load_inference_streaming(dir.path()).unwrap();
    assert!(streamed.is_streaming_bundle("codec.safetensors"));
    assert!(streamed.is_streaming_bundle("generator.safetensors"));
    assert!(!streamed.is_streaming_bundle("training.safetensors"));
    assert!(streamed.bundle_f32_weights_verified_finite("codec.safetensors"));
    assert!(streamed.bundle_f32_weights_verified_finite("generator.safetensors"));
    assert!(streamed.file("codec.safetensors").is_err());
    assert!(streamed.file("generator.safetensors").is_err());
    assert!(streamed.file("training.safetensors").is_err());
    assert_eq!(streamed.file("config.json").unwrap(), b"{}");

    for bundle in ["codec.safetensors", "generator.safetensors"] {
        let mut from_copied = BTreeMap::new();
        copied
            .visit_bundle_tensors(
                bundle,
                |e| e.to_string(),
                |name, shape, dtype, data| {
                    from_copied.insert(name.to_owned(), (shape.to_vec(), dtype, data.to_vec()));
                    Ok(())
                },
            )
            .unwrap();
        let mut from_streamed = BTreeMap::new();
        streamed
            .visit_bundle_tensors(
                bundle,
                |e| e.to_string(),
                |name, shape, dtype, data| {
                    from_streamed.insert(name.to_owned(), (shape.to_vec(), dtype, data.to_vec()));
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(from_copied, from_streamed);
    }
    assert_eq!(
        streamed
            .read_bundle_tensor("generator.safetensors", "attn_mask_for_masking")
            .unwrap(),
        mask_raw
    );
    streamed.release_file("codec.safetensors").unwrap();
    assert!(!streamed.is_streaming_bundle("codec.safetensors"));
    assert!(streamed.release_file("codec.safetensors").is_err());
}

#[cfg(unix)]
#[test]
fn streaming_inference_rejects_tampering_nonfinite_weights_and_symlinks() {
    // 1. Nonfinite weight in codec (small path) is rejected even when hashes match.
    let (dir, mut manifest) = fixture();
    let bad: Vec<u8> = [1.0f32, f32::NAN]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let bytes = serialize(
        [("x", TensorView::new(Dtype::F32, vec![2], &bad).unwrap())],
        None,
    )
    .unwrap();
    fs::write(dir.path().join("codec.safetensors"), &bytes).unwrap();
    manifest.files.insert(
        "codec.safetensors".into(),
        FileRecord {
            bytes: bytes.len() as u64,
            sha256: sha256(&bytes),
        },
    );
    manifest.tensors.insert(
        "codec.safetensors".into(),
        vec![TensorRecord {
            name: "x".into(),
            shape: vec![2],
            dtype: TensorDtype::F32,
            bytes: bad.len(),
            sha256: sha256(&bad),
        }],
    );
    save(dir.path(), &manifest);
    assert!(Artifact::load_inference_streaming(dir.path()).is_err());

    // 2. Nonfinite weight in parallel (>= 4 MiB) bundle is rejected even when hashes match.
    let (dir, mut manifest) = fixture();
    let mut big = vec![0u8; 4_400_000];
    big[4_399_996..4_400_000].copy_from_slice(&f32::INFINITY.to_le_bytes());
    let bytes = serialize(
        [(
            "w",
            TensorView::new(Dtype::F32, vec![1_100_000], &big).unwrap(),
        )],
        None,
    )
    .unwrap();
    fs::write(dir.path().join("generator.safetensors"), &bytes).unwrap();
    manifest.files.insert(
        "generator.safetensors".into(),
        FileRecord {
            bytes: bytes.len() as u64,
            sha256: sha256(&bytes),
        },
    );
    manifest.tensors.insert(
        "generator.safetensors".into(),
        vec![TensorRecord {
            name: "w".into(),
            shape: vec![1_100_000],
            dtype: TensorDtype::F32,
            bytes: big.len(),
            sha256: sha256(&big),
        }],
    );
    save(dir.path(), &manifest);
    assert!(Artifact::load_inference_streaming(dir.path()).is_err());

    // 3. Matching outer file hash does not bypass per-tensor hash check.
    let (dir, mut manifest) = fixture();
    let path = dir.path().join("generator.safetensors");
    let mut raw = fs::read(&path).unwrap();
    *raw.last_mut().unwrap() ^= 1;
    fs::write(&path, &raw).unwrap();
    manifest
        .files
        .get_mut("generator.safetensors")
        .unwrap()
        .sha256 = sha256(&raw);
    save(dir.path(), &manifest);
    assert!(Artifact::load_inference_streaming(dir.path()).is_err());

    // 4. Symlink bundle is rejected via O_NOFOLLOW.
    let (dir, _) = fixture();
    fs::rename(
        dir.path().join("codec.safetensors"),
        dir.path().join("real.safetensors"),
    )
    .unwrap();
    std::os::unix::fs::symlink("real.safetensors", dir.path().join("codec.safetensors")).unwrap();
    assert!(Artifact::load_inference_streaming(dir.path()).is_err());
}

#[test]
fn local_ncm_v1_profile_admits_minimal_and_optional_files_and_rejects_extras() {
    let (dir, mut manifest) = fixture();
    manifest.profile = "local-ncm-v1".into();
    manifest.upstream_commit = "local-dev".into();
    manifest.checkpoint_repository = "local/my-ncm".into();
    manifest.checkpoint_revision = "v1".into();
    // Drop optional training.safetensors and tokenizer.json to test minimal 4-file artifact.
    manifest.files.remove("training.safetensors");
    manifest.tensors.remove("training.safetensors");
    manifest.files.remove("tokenizer.json");
    save(dir.path(), &manifest);

    let loaded = Artifact::load_inference(dir.path()).unwrap();
    assert_eq!(loaded.manifest().profile, "local-ncm-v1");
    assert!(loaded.file("tokenizer.json").is_err());

    #[cfg(unix)]
    {
        let streamed = Artifact::load_inference_streaming(dir.path()).unwrap();
        assert_eq!(streamed.manifest().profile, "local-ncm-v1");
        assert!(streamed.is_streaming_bundle("codec.safetensors"));
        assert!(streamed.is_streaming_bundle("generator.safetensors"));
    }

    // Adding an undeclared file name must fail closed.
    let rogue = b"nope";
    fs::write(dir.path().join("extra.bin"), rogue).unwrap();
    manifest.files.insert(
        "extra.bin".into(),
        FileRecord {
            bytes: rogue.len() as u64,
            sha256: sha256(rogue),
        },
    );
    save(dir.path(), &manifest);
    assert!(Artifact::load_inference(dir.path()).is_err());
    manifest.files.remove("extra.bin");

    // Empty provenance fields must be rejected.
    manifest.checkpoint_repository = String::new();
    save(dir.path(), &manifest);
    assert!(Artifact::load_inference(dir.path()).is_err());
}

#[cfg(unix)]
#[test]
fn prequant_sidecar_roundtrip_fallback_and_tamper_rejection() {
    use ncm_io::{
        PrequantKind, PrequantTensorView, PrequantWriteEntry, f32_bytes_to_bf16_bytes,
        write_prequant_sidecar,
    };

    let (dir, manifest) = fixture();

    // 1. Absent sidecar falls back cleanly to standard streaming load.
    let fallback = Artifact::load_inference_with_prequant(dir.path(), "ggml", "q8").unwrap();
    assert!(!fallback.has_prequant());
    assert!(fallback.is_streaming_bundle("codec.safetensors"));

    // 2. Write valid prequant sidecar for codec + generator.
    let mut entries = Vec::new();
    for bundle in ["codec.safetensors", "generator.safetensors"] {
        fallback
            .visit_bundle_tensors(
                bundle,
                |e| e,
                |name, shape, dtype, data| {
                    let key = format!("{bundle}/{name}");
                    let (kind, st_dtype, st_shape, bytes) = match dtype {
                        TensorDtype::I64 => (
                            PrequantKind::I64,
                            safetensors::Dtype::I64,
                            shape.to_vec(),
                            data.to_vec(),
                        ),
                        TensorDtype::F32 if name == "attn_mask_for_masking" => (
                            PrequantKind::F32,
                            safetensors::Dtype::F32,
                            shape.to_vec(),
                            data.to_vec(),
                        ),
                        TensorDtype::F32 => {
                            let bf = f32_bytes_to_bf16_bytes(data)?;
                            (
                                PrequantKind::Bf16,
                                safetensors::Dtype::BF16,
                                shape.to_vec(),
                                bf,
                            )
                        }
                    };
                    entries.push(PrequantWriteEntry {
                        bundle: bundle.to_owned(),
                        name: name.to_owned(),
                        shape: shape.to_vec(),
                        kind,
                        parts: vec![(key, st_dtype, st_shape, bytes)],
                    });
                    Ok(())
                },
            )
            .unwrap();
    }
    let summary = write_prequant_sidecar(dir.path(), "ggml", "q8", entries, false).unwrap();
    assert!(summary.file_bytes > 0);
    assert!(summary.manifest_path.is_file());
    assert!(summary.weights_path.is_file());

    // Parent streaming load remains completely unaffected.
    assert!(
        !Artifact::load_inference_streaming(dir.path())
            .unwrap()
            .has_prequant()
    );

    // Loading with prequant picks up the verified sidecar.
    let loaded = Artifact::load_inference_with_prequant(dir.path(), "ggml", "q8").unwrap();
    assert!(loaded.has_prequant());
    assert!(loaded.has_prequant_bundle("codec.safetensors"));
    assert!(loaded.has_prequant_bundle("generator.safetensors"));
    let mut visited = 0usize;
    loaded
        .visit_bundle_prequant(
            "generator.safetensors",
            |e| e,
            |_name, _shape, view| {
                assert!(matches!(
                    view,
                    PrequantTensorView::Bf16(_)
                        | PrequantTensorView::F32(_)
                        | PrequantTensorView::I64(_)
                ));
                visited += 1;
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(visited, manifest.tensors["generator.safetensors"].len());

    // 3. Tampering 1 byte in weights safetensors must fail closed (never fall back).
    let weights_path = dir.path().join("prequant-ggml-q8.safetensors");
    let orig_weights = fs::read(&weights_path).unwrap();
    let mut tampered = orig_weights.clone();
    *tampered.last_mut().unwrap() ^= 0xff;
    fs::write(&weights_path, &tampered).unwrap();
    assert!(Artifact::load_inference_with_prequant(dir.path(), "ggml", "q8").is_err());
    fs::write(&weights_path, &orig_weights).unwrap();
    assert!(Artifact::load_inference_with_prequant(dir.path(), "ggml", "q8").is_ok());

    // 4. Stale manifest_sha256 (mutated manifest.json whitespace) must fail closed.
    let manifest_path = dir.path().join("manifest.json");
    let orig_manifest = fs::read(&manifest_path).unwrap();
    let mut altered_manifest = orig_manifest.clone();
    altered_manifest.extend_from_slice(b"\n");
    fs::write(&manifest_path, &altered_manifest).unwrap();
    assert!(Artifact::load_inference_with_prequant(dir.path(), "ggml", "q8").is_err());
    fs::write(&manifest_path, &orig_manifest).unwrap();

    // 5. Incomplete pair (missing .json) must fail closed.
    let json_path = dir.path().join("prequant-ggml-q8.json");
    let orig_json = fs::read(&json_path).unwrap();
    fs::remove_file(&json_path).unwrap();
    assert!(Artifact::load_inference_with_prequant(dir.path(), "ggml", "q8").is_err());
    fs::write(&json_path, &orig_json).unwrap();

    // 6. Symlinked sidecar file must fail closed.
    fs::rename(&weights_path, dir.path().join("real-pq.safetensors")).unwrap();
    std::os::unix::fs::symlink("real-pq.safetensors", &weights_path).unwrap();
    assert!(Artifact::load_inference_with_prequant(dir.path(), "ggml", "q8").is_err());
}
