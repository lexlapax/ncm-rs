"""Reusable bounded current-upstream inspection; no launch or hardware access."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from readonly_trace_preflight import file_identity, require
from readonly_trace_runtime import metadata_probe
from prepare_clock_python import UPSTREAM as UPSTREAM_COMMIT


def inspect(root, tree, output, *, git_path, home, config_sha):
    """Inspect a supplied standalone checkout; caller binds tree/config to review.

    Fresh private output; no code import from the checkout and no Git mutation.
    This snapshot is not atomic, import-closure proof, or launch authorization.
    """
    root, tree, output, git_path, home = map(Path, (root, tree, output, git_path, home))
    require(all(p.is_absolute() for p in (root, tree, output, git_path, home))
            and root.resolve(strict=True) == root and tree.parent.resolve(strict=True) == tree.parent
            and output.parent.resolve(strict=True) == output.parent
            and git_path.resolve(strict=True) == git_path and home.resolve(strict=True) == home,
            'canonical absolute preparation paths required')
    require(isinstance(config_sha, str) and re.fullmatch(r'[0-9a-f]{64}', config_sha),
            'explicit reviewed Git configuration hash required')
    require(not os.path.lexists(output) and not output.is_relative_to(tree),
            'fresh output outside upstream tree required')
    env = {'PATH': os.defpath, 'HOME': str(home), 'LANG': 'C', 'LC_ALL': 'C', 'PYTHONNOUSERSITE': '1',
           'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_OPTIONAL_LOCKS': '0',
           'GIT_TERMINAL_PROMPT': '0'}
    require(tree.resolve(strict=True)==tree and (tree/'.git').is_dir(),'canonical standalone upstream required')
    output.mkdir(mode=0o700)
    config=file_identity(tree/'.git/config',config_sha)
    git=file_identity(git_path); index=file_identity(tree/'.git/index')
    receipts={}
    def probe(name,args):
        text,receipt=metadata_probe([str(git_path),'--no-replace-objects','--no-optional-locks','-c','core.fsmonitor=false',
            '-c','core.untrackedCache=false','-C',str(tree),*args],root,env,output,name)
        receipts[name]=receipt
        return text

    def nul(text):
        require(not text or text.endswith('\0'),'truncated Git listing')
        return text.split('\0')[:-1] if text else []

    require(probe('head-before',['rev-parse','HEAD']).strip()==UPSTREAM_COMMIT,'upstream HEAD differs')
    probe('raw-commit',['cat-file','commit',UPSTREAM_COMMIT])
    commit=(output/'raw-commit.txt').read_bytes()
    require(hashlib.sha1(b'commit '+str(len(commit)).encode()+b'\0'+commit).hexdigest()==UPSTREAM_COMMIT,
            'raw commit object differs from pinned SHA-1')
    require(probe('top-level',['rev-parse','--show-toplevel']).strip()==str(tree),'unexpected upstream worktree root')
    require(probe('status-before',['status','--porcelain=v1','-z','--untracked-files=all'])=='','upstream dirty/untracked')
    flags=nul(probe('index-flags',['ls-files','-v','-z']))
    require(flags and len(flags)<=512 and all(row.startswith('H ') for row in flags),'hidden or unsupported index flags')
    flag_paths=[row[2:] for row in flags]
    head=nul(probe('head-tree',['ls-tree','-r','-z','--full-tree',UPSTREAM_COMMIT]))
    tracked={}
    for row in head:
        prefix,name=row.split('\t',1); mode,kind,oid=prefix.split()
        path=tree/name
        require(mode in ('100644','100755') and kind=='blob' and len(oid)==40
                and name not in tracked and path.is_relative_to(tree) and '..' not in Path(name).parts
                and path.resolve(strict=True)==path,'unsupported upstream source entry')
        tracked[name]=(mode,oid)
    require(set(tracked)==set(flag_paths) and len(flag_paths)==len(tracked),'index/HEAD path sets differ')
    staged=nul(probe('index-entries',['ls-files','--stage','-z']))
    expected=[]
    for row in staged:
        prefix,name=row.split('\t',1); mode,oid,stage=prefix.split()
        require(stage=='0' and tracked.get(name)==(mode,oid),'index differs from pinned tree')
        expected.append(name)
    require(set(expected)==set(tracked) and len(expected)==len(tracked),'index entries differ')
    ignored=nul(probe('ignored',['ls-files','--others','--ignored','--exclude-standard','-z']))
    inventory=[]; total=0
    for name in sorted(tracked):
        path=tree/name; current=path.stat(follow_symlinks=False)
        require(stat.S_ISREG(current.st_mode) and current.st_size<=8*1024**2,'nonregular/oversized upstream source')
        total+=current.st_size; require(total<=64*1024**2,'upstream source inventory exceeds bound')
        inventory.append((name,current))
    files=[]
    for name,before in inventory:
        path=tree/name; identity=file_identity(path,size=before.st_size)
        require((identity['device'],identity['inode'])==(before.st_dev,before.st_ino),'source replaced while hashing')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as source:
            current=os.fstat(source.fileno())
            require(stat.S_ISREG(current.st_mode) and (current.st_dev,current.st_ino)==(before.st_dev,before.st_ino),
                    'source identity changed before Git blob check')
            raw=source.read(before.st_size+1)
        require(len(raw)==before.st_size and hashlib.sha256(raw).hexdigest()==identity['sha256'],'source changed between reads')
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        require(blob==tracked[name][1],'upstream source bytes differ from pinned Git blob')
        require(bool(before.st_mode & 0o111)==(tracked[name][0]=='100755'),'upstream source executable mode differs')
        files.append(dict(relative_path=name,git_mode=tracked[name][0],git_blob_sha1=blob,identity=identity))
    require(probe('head-after',['rev-parse','HEAD']).strip()==UPSTREAM_COMMIT,'upstream HEAD changed')
    require(probe('status-after',['status','--porcelain=v1','-z','--untracked-files=all'])=='','upstream changed/dirty')
    for identity in (git,config,index):
        current=file_identity(identity['path'],identity['sha256'],identity['bytes'])
        require(all(current[k]==identity[k] for k in ('path','sha256','bytes','device','inode')),
                'Git utility/config/index changed')
    for name,before in inventory:
        path=tree/name; current=path.stat(follow_symlinks=False)
        require(path.resolve(strict=True)==path and all(getattr(before,k)==getattr(current,k) for k in
            ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_mode')),'source changed during inspection')
    result=dict(schema_version=1,current_upstream_snapshot_verified=True,upstream_commit=UPSTREAM_COMMIT,
        tracked_clean=True,untracked_count=0,tracked_file_count=len(files),tracked_bytes=total,
        replacement_objects_disabled=True,raw_commit_sha1_verified=True,
        ignored_entry_count=len(ignored),ignored_entries_inspected=False,git_identity=git,config_identity=config,
        index_identity=index,files=files,receipts=receipts,launch_ready=False,python_import_closure_verified=False,
        historical_interpreter_library_bytes_verified=False,atomic_snapshot_verified=False,
        scope='Current pinned Git tree/index and tracked regular-file bytes only. Private raw Git listings retained. '
              'Ignored entries are counted, not qualified; no source import, Python bytecode/native closure, '
              'atomic snapshot, benchmark or launch-readiness proof. Recheck during fresh full preparation.')
    require(not ignored, 'ignored upstream entries require separate review')
    fd = os.open(output/'upstream.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return result
