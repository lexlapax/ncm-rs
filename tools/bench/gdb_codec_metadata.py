"""GDB-only diagnostic for Rust formatting/allocation work in codec_profile.

Launch an owned trusted local executable, never attach to an unrelated process.
NCM_METADATA_SYMBOLS supplies nm-derived raw/demangled symbols; NCM_METADATA_REPORT
is a new output file. Breakpoint counts are NOT timing/performance measurements.
Allocator entry ABI decoding is explicitly Linux x86-64 SysV. libc entries also
cover Rust allocations whose allocator shims were inlined; native calls count too.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import gdb

symbols_path = Path(os.environ['NCM_METADATA_SYMBOLS'])
symbols = json.loads(symbols_path.read_text())
output = Path(os.environ['NCM_METADATA_REPORT'])
if output.exists():
    raise ValueError('output exists')
with Path(gdb.current_progspace().filename).open('rb') as file:
    if hashlib.file_digest(file, 'sha256').hexdigest() != symbols['binary_sha256']:
        raise ValueError('symbol map belongs to a different executable')
raw_names = symbols['raw_to_demangled']
active_formats = collections.defaultdict(list)
active_scopes = collections.defaultdict(list)
formats = collections.defaultdict(lambda: {'calls': 0, 'allocations': 0, 'reallocations': 0, 'requested_bytes': 0})
scopes = collections.defaultdict(lambda: {'calls': 0, 'allocations': 0, 'reallocations': 0, 'requested_bytes': 0})
scope_instances = collections.defaultdict(list)
all_calls = {'allocations': 0, 'reallocations': 0, 'requested_bytes': 0}
errors = []


def thread():
    return gdb.selected_thread().global_num


class Leave(gdb.FinishBreakpoint):
    def __init__(self, frame, stack, key):
        super().__init__(frame, internal=True)
        self.stack, self.key = stack, key

    def stop(self):
        if not self.stack or self.stack.pop() != self.key:
            errors.append('unbalanced instrumentation scopes')
        return False

    def out_of_scope(self):
        errors.append('instrumented frame unwound or exited without return')


class EnterFormat(gdb.Breakpoint):
    def stop(self):
        frame = gdb.newest_frame()
        caller = frame.older()
        if caller is None:
            errors.append('missing format caller')
            return False
        raw = caller.name() or '<unknown>'
        name = raw_names.get(raw, raw)
        if 'ncm_model::' in name or 'ncm_model..' in name:
            formats[name]['calls'] += 1
            stack = active_formats[thread()]
            stack.append(name)
            Leave(frame, stack, name)
        return False


class EnterScope(gdb.Breakpoint):
    def __init__(self, raw, name):
        super().__init__(raw, internal=True)
        self.name = name

    def stop(self):
        scopes[self.name]['calls'] += 1
        key = (self.name, len(scope_instances[self.name]))
        scope_instances[self.name].append({'allocations': 0, 'reallocations': 0, 'requested_bytes': 0})
        stack = active_scopes[thread()]
        stack.append(key)
        Leave(gdb.newest_frame(), stack, key)
        return False


class Allocate(gdb.Breakpoint):
    def __init__(self, raw, size_expression, realloc=False):
        super().__init__(raw, internal=True)
        self.realloc = realloc
        self.size_expression = size_expression

    def stop(self):
        if gdb.newest_frame().architecture().name() != 'i386:x86-64':
            raise RuntimeError('unsupported register ABI')
        # Entry requests, not successful allocations, distinct logical objects,
        # or RSS. Expressions below follow the Linux x86-64 SysV libc ABI.
        size = int(gdb.parse_and_eval(self.size_expression))
        key = 'reallocations' if self.realloc else 'allocations'
        targets = [all_calls]
        if active_formats[thread()]:
            targets.append(formats[active_formats[thread()][-1]])
        if active_scopes[thread()]:
            name, index = active_scopes[thread()][-1]
            targets.extend([scopes[name], scope_instances[name][index]])
        for target in targets:
            target[key] += 1
            target['requested_bytes'] += size
        return False


EnterFormat(symbols['format'], internal=True)
gdb.execute('set breakpoint pending on')
for name, expression, realloc in [('malloc', '$rdi', False), ('calloc', '$rdi * $rsi', False),
                                  ('realloc', '$rsi', True), ('posix_memalign', '$rdx', False)]:
    Allocate(name, expression, realloc)
for key in ('encode', 'block'):
    EnterScope(symbols[key], key)


def exited(event):
    if any(active_formats.values()) or any(active_scopes.values()):
        errors.append('nonempty scope stacks at exit')
    if getattr(event, 'exit_code', None) != 0:
        errors.append('inferior did not exit successfully')
    if not formats or scopes['encode']['calls'] != 2 or scopes['block']['calls'] != 2:
        errors.append('not the expected two-reconstruction profile or missing metadata attribution')
    executable = Path(gdb.current_progspace().filename)
    with executable.open('rb') as file:
        binary_hash = hashlib.file_digest(file, 'sha256').hexdigest()
    report = {'schema_version': 1, 'diagnostic_profiling': True, 'all_checks_passed': not errors,
              'errors': errors, 'binary_sha256': binary_hash,
              'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'symbol_map_sha256': hashlib.sha256(symbols_path.read_bytes()).hexdigest(),
              'scope': 'Checked load plus two B1 reconstructions, with GDB breakpoints; no timing claims.',
              'model_format_callers': dict(formats), 'execution_scopes': dict(scopes),
              'scope_invocations_in_order': dict(scope_instances),
              'whole_process_allocator_entry_requests': all_calls,
              'limitations': ['Formatting counts cover direct model callers of the identified formatter symbol, not all possible formatting implementations.',
                             'malloc/calloc/realloc/posix_memalign entries include Rust and native-library calls, and possibly nested allocator entries. They are not distinct successful allocations.',
                             'Direct model formatter-associated entries are attributed separately; custom allocators and other entry points are not covered.',
                             'No deallocation, live/peak memory, allocation-success or elapsed-time attribution.',
                             'No debug symbols or production source changes; nm symbol identity and Linux x86-64 SysV ABI are explicit in this capture.']}
    with output.open('x') as file:
        json.dump(report, file, indent=2); file.write('\n')


gdb.events.exited.connect(exited)
gdb.execute('run')
