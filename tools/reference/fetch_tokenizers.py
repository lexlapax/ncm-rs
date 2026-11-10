"""Fetch only explicitly pinned tokenizer/config/license assets; never model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = {
    "tinystories": ("hazyresearch/ncm-checkpoints", "d342ce8aaea58830eef3b8ad473664673e41f692", "mit",
        ["tinystories/bpe_vocab_8192.pickle", "tinystories/bpe_merges_8192.pickle", "tinystories/config.yaml", "tinystories/README.md", "README.md"]),
    "gpt2": ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e", "mit",
        ["tokenizer.json", "tokenizer_config.json", "config.json", "vocab.json", "merges.txt", "README.md"]),
    "qwen3": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "apache-2.0",
        ["tokenizer.json", "tokenizer_config.json", "config.json", "vocab.json", "merges.txt", "LICENSE", "README.md"]),
    "bert": ("google-bert/bert-base-uncased", "86b5e0934494bd15c9632b12f734a8a67f723594", "apache-2.0",
        ["tokenizer.json", "tokenizer_config.json", "config.json", "vocab.txt", "LICENSE", "README.md"]),
}
MAX_FILE = 32 * 1024 * 1024


def download(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        data = response.read(MAX_FILE+1)
    if len(data) > MAX_FILE:
        raise ValueError("tokenizer asset exceeds size bound")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--lock", type=Path, help="require exact previously reviewed manifest")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must not exist")
    args.output.mkdir(parents=True)
    manifest = {"schema_version": 1, "sources": {}}
    for profile, (repo, revision, license_id, files) in SOURCES.items():
        record = {"repository": repo, "revision": revision, "license": license_id, "files": {}}
        trees = {}
        for name in files:
            parent = name.rpartition("/")[0]
            if parent not in trees:
                url = f"https://huggingface.co/api/models/{repo}/tree/{revision}" + ("/"+parent if parent else "")
                trees[parent] = {item["path"]: item for item in json.loads(download(url))}
            info = trees[parent][name]
            if info["type"] != "file" or info["size"] > MAX_FILE:
                raise ValueError("invalid source file metadata")
            data = download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}")
            sha = hashlib.sha256(data).hexdigest()
            if len(data) != info["size"]:
                raise ValueError("source size mismatch")
            if "lfs" in info:
                if sha != info["lfs"]["oid"]:
                    raise ValueError("LFS digest mismatch")
            elif hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest() != info["oid"]:
                raise ValueError("Git blob identity mismatch")
            path = args.output / profile / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            record["files"][name] = {"bytes":len(data), "sha256":sha, "git_oid":info["oid"]}
            print(profile, name, len(data), sha)
        if profile == "gpt2":
            url = "https://raw.githubusercontent.com/openai/gpt-2/9b63575ef42771a015060c964af2c3da4cf7c8ab/LICENSE"
            data = download(url)
            oid = hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            if len(data) != 1403 or oid != "f56abfef079f3850e93db8bd6ee32582c3e65956":
                raise ValueError("GPT-2 license identity mismatch")
            (args.output/profile/"OPENAI_LICENSE.txt").write_bytes(data)
            record["files"]["OPENAI_LICENSE.txt"] = {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"git_oid":oid,"url":url}
        manifest["sources"][profile] = record
    if args.lock and json.loads(args.lock.read_bytes()) != manifest:
        raise ValueError("downloaded assets differ from reviewed lock")
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
