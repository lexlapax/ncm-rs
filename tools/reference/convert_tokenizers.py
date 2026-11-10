"""Bounded data-only TinyStories conversion and offline pinned HF wrapper oracles."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickletools
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "c70274273f489abc73832b2b98584a8483b6b203"
VERSIONS = {"transformers":"4.57.6", "tokenizers":"0.22.2", "regex":"2025.9.18"}
SPECIAL = {"gpt2": {"eos_token":"<|endoftext|>","pad_token":"<|pad|>"},
           "qwen3": {"eos_token":"<|im_end|>","pad_token":"<|endoftext|>"},
           "bert": {"eos_token":"[CLS]","pad_token":"[PAD]"}}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)+"\n").encode()


def data_pickle(data):
    """Interpret ONLY the protocol-4 data opcodes present in the pinned assets.

    No pickle.load/Unpickler, imports, globals, object constructors or callable
    invocation. Mutable-container cycles cannot pass the subsequent fixed schema.
    """
    if len(data) > 2*1024*1024:
        raise ValueError("pickle exceeds input bound")
    stack, memo = [], []
    mark = object()
    frame_end = None
    stopped = False
    for count, (op, arg, position) in enumerate(pickletools.genops(data)):
        if count > 100000 or len(stack) > 32768 or len(memo) > 32768:
            raise ValueError("pickle complexity bound")
        name = op.name
        if count == 0:
            if name != "PROTO" or arg != 4:
                raise ValueError("only protocol four is supported")
            continue
        if name == "FRAME":
            if (frame_end is not None and frame_end != position) or arg <= 0 or position+9+arg > len(data):
                raise ValueError("invalid pickle frame")
            frame_end = position+9+arg
            continue
        if frame_end is None or position >= frame_end:
            raise ValueError("opcode outside bounded frame")
        if name == "EMPTY_DICT": stack.append({})
        elif name == "EMPTY_LIST": stack.append([])
        elif name == "MARK": stack.append(mark)
        elif name in ("BININT1", "BININT2", "SHORT_BINBYTES"): stack.append(arg)
        elif name == "MEMOIZE": memo.append(stack[-1])
        elif name in ("BINGET", "LONG_BINGET"):
            if not 0 <= arg < len(memo): raise ValueError("invalid memo reference")
            stack.append(memo[arg])
        elif name == "TUPLE2":
            right = stack.pop(); left = stack.pop(); stack.append((left,right))
        elif name in ("APPENDS", "SETITEMS"):
            split = len(stack)-1
            while split >= 0 and stack[split] is not mark: split -= 1
            if split < 1: raise ValueError("missing container mark")
            values = stack[split+1:]; del stack[split:]
            container = stack[-1]
            if name == "APPENDS":
                if type(container) is not list: raise ValueError("expected list")
                container.extend(values)
            else:
                if type(container) is not dict or len(values)%2: raise ValueError("expected key/value pairs")
                for key,value in zip(values[::2],values[1::2]):
                    if type(key) is not int or key in container: raise ValueError("invalid/duplicate vocabulary key")
                    container[key] = value
        elif name == "STOP":
            if len(stack) != 1 or position+1 != len(data) or frame_end != len(data):
                raise ValueError("invalid stop or trailing content")
            stopped = True
        else:
            raise ValueError(f"unsupported pickle opcode: {name}")
    if not stopped: raise ValueError("missing STOP")
    return stack[0]


def validate_bpe(vocab, merges):
    if type(vocab) is not dict or len(vocab) != 8192 or set(vocab) != set(range(8192)):
        raise ValueError("invalid pinned vocabulary inventory")
    if any(type(v) is not bytes or not v or len(v)>4096 for v in vocab.values()):
        raise ValueError("vocabulary must contain bounded byte strings")
    if len(set(vocab.values())) != len(vocab): raise ValueError("duplicate vocabulary bytes")
    if vocab[0] != b"<|endoftext|>" or vocab[1] != b"<|pad|>" or any(vocab[i+2] != bytes([i]) for i in range(256)):
        raise ValueError("special/byte token IDs differ")
    if type(merges) is not list or len(merges) != 7934:
        raise ValueError("invalid pinned merge inventory")
    seen, known = set(), set(vocab[i] for i in range(258))
    for rank, pair in enumerate(merges):
        if type(pair) is not tuple or len(pair)!=2 or any(type(x) is not bytes for x in pair):
            raise ValueError("merges require byte pairs")
        if pair in seen or any(x not in known for x in pair) or pair[0]+pair[1] != vocab[258+rank]:
            raise ValueError("inconsistent ranked merges")
        seen.add(pair); known.add(pair[0]+pair[1])


def texts():
    samples = ["", "hello world", " Hello  world ", "can't won't I'm I'M we've THEY'RE", "a... b?! (c), d.",
        "\t\n\r\n  trailing   ", "one\u00a0two\u2003three\u0085four\x1cfive", "café cafe\u0301 naïve Straße Ελληνικά",
        "中文 日本語 한국어 العربية עברית", "👩🏽‍💻🙂🏳️‍🌈", "\x00\x01\x7f", "1234567890 ١٢٣ １２３", "\u1c89\U000105c0",
        "<|endoftext|><|pad|>", "x<|pad|>y<|endoftext|>z", "[CLS]Hi[MASK]there[SEP][PAD]", "<|im_start|>user\nhi<|im_end|>",
        "<think>why?</think>", "foo . bar ? baz ! don't ' quote ", "".join(chr(i) for i in range(128)),
        "e\u0301\u0308\u200d\ufe0f", " aaaa aaaaaa ababababab", "\n\n\n", "  ", "<|endoftext|><|endoftext|>"]
    rng = random.Random(42)
    alphabet = list("abcXYZ0123 '.!?\t\r\n") + ["é", "中", "🙂", "\u0301", "\u00a0", "\u2003"]
    samples += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1,60))) for _ in range(64)]
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path); parser.add_argument("upstream", type=Path); parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("output must not exist")
    for package, wanted in VERSIONS.items():
        if importlib.metadata.version(package) != wanted: raise ValueError(f"unpinned {package}")
    if subprocess.check_output(["git","-C",str(args.upstream),"rev-parse","HEAD"],text=True).strip()!=UPSTREAM:
        raise ValueError("unpinned source checkout")
    if subprocess.check_output(["git","-C",str(args.upstream),"status","--porcelain"],text=True).strip():
        raise ValueError("dirty upstream checkout")
    lock = json.loads((ROOT/"tools/reference/tokenizer-sources.lock.json").read_bytes())
    for profile, record in lock["sources"].items():
        for name, expected in record["files"].items():
            path = args.source/profile/name
            if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"]:
                raise ValueError("invalid source file")
            if digest(path.read_bytes()) != expected["sha256"]: raise ValueError("source hash mismatch")
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(args.upstream.resolve()))
    from tokenization.bpe_tokenizer import BPETokenizer
    from tokenization.hf_tokenizer import HFTokenizer
    vocab = data_pickle((args.source/"tinystories/tinystories/bpe_vocab_8192.pickle").read_bytes())
    merges = data_pickle((args.source/"tinystories/tinystories/bpe_merges_8192.pickle").read_bytes())
    validate_bpe(vocab,merges)
    tiny = BPETokenizer(vocab,merges,{"eos_token":"<|endoftext|>","pad_token":"<|pad|>"})
    args.output.mkdir(parents=True)
    fixtures, output_lock = {}, {"schema_version":1,"profiles":{}}
    for profile in lock["sources"]:
        if profile == "tinystories":
            tokenizer = tiny
            payload = {"schema_version":1,"vocab_hex":[vocab[i].hex() for i in range(len(vocab))],
                       "merges_hex":[[a.hex(),b.hex()] for a,b in merges]}
            kind, total, cleanup = "byte_bpe", len(vocab), False
        else:
            tokenizer = HFTokenizer(str((args.source/profile).resolve()),local_files_only=True,trust_remote_code=False,use_fast=True,**SPECIAL[profile])
            if not tokenizer.hf_tokenizer.is_fast: raise ValueError("slow HF fallback is unsupported")
            payload = json.loads(tokenizer.hf_tokenizer.backend_tokenizer.to_str())
            kind, total, cleanup = "huggingface", len(tokenizer.hf_tokenizer), tokenizer.hf_tokenizer.clean_up_tokenization_spaces
        directory = args.output/profile; directory.mkdir()
        data = json_bytes(payload); (directory/"tokenizer.json").write_bytes(data)
        license_path = args.upstream/"LICENSE" if profile == "tinystories" else args.source/profile/("OPENAI_LICENSE.txt" if profile == "gpt2" else "LICENSE")
        license_bytes = license_path.read_bytes()
        (directory/"LICENSE").write_bytes(license_bytes)
        metadata = {"schema_version":1,"profile":profile,"kind":kind,"upstream_commit":UPSTREAM,
                    "source":lock["sources"][profile],"reference_versions":VERSIONS,
                    "vocab_size":tokenizer.vocab_size,"total_vocab_size":total,
                    "eos_token_id":tokenizer.eos_token_id,"pad_token_id":tokenizer.pad_token_id,
                    "cleanup":cleanup,"payload_bytes":len(data),"payload_sha256":digest(data),"license_sha256":digest(license_bytes)}
        manifest = json_bytes(metadata); (directory/"manifest.json").write_bytes(manifest)
        output_lock["profiles"][profile] = {"manifest_sha256":digest(manifest),"payload_sha256":digest(data)}
        cases = []
        for text in texts():
            ids = tokenizer.encode(text)
            cases.append({"text":text,"ids":ids,"decoded":tokenizer.decode(ids),"decoded_skip_special":tokenizer.decode(ids,True)})
        decode_ids = [[tokenizer.eos_token_id,tokenizer.pad_token_id],[]]
        if profile == "tinystories":
            decode_ids += [[i+2] for i in range(256)]
            decode_ids += [[x+2 for x in seq] for seq in [[0xf0,0x28,0x8c,0x28],[0xc0,0xaf],[0xed,0xa0,0x80],[0xe2,0x82]]]
        else:
            decode_ids += [[x] for x in tokenizer.hf_tokenizer.all_special_ids]
            if profile != "bert":
                from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
                mapping = bytes_to_unicode()
                decode_ids += [[tokenizer.hf_tokenizer.convert_tokens_to_ids(mapping[i])] for i in range(256)]
        fixtures[profile] = {"manifest_sha256":digest(manifest),"vocab_size":tokenizer.vocab_size,"total_vocab_size":total,
            "eos_token_id":tokenizer.eos_token_id,"pad_token_id":tokenizer.pad_token_id,"cleanup":cleanup,"cases":cases,
            "decode_cases":[{"ids":ids,"decoded":tokenizer.decode(ids),"decoded_skip_special":tokenizer.decode(ids,True)} for ids in decode_ids]}
        print(profile,kind,tokenizer.vocab_size,total,tokenizer.eos_token_id,tokenizer.pad_token_id,"cleanup",cleanup,len(cases),len(decode_ids))
    (args.output/"lock.json").write_bytes(json_bytes(output_lock))
    (args.output/"fixture.json").write_bytes(json_bytes({"schema_version":1,"upstream_commit":UPSTREAM,"reference_versions":VERSIONS,
        "script_sha256":digest(Path(__file__).read_bytes()),"unicode_policy":"Python regex and Rust regex-syntax Unicode 16.0 for TinyStories",
        "profiles":fixtures}))


if __name__ == "__main__":
    main()
