"""Read-only NVIDIA identity and supported-clock metadata; never changes clocks.

This component does not check execution privileges or prove that application-clock
list entries can be locked/enforced. It is not quiet-window or launch readiness.
"""
import csv
import io
import os
from pathlib import Path
import re

from prepare_readonly_launch import parse_device
from readonly_trace_preflight import file_identity, require
from readonly_trace_runtime import clean_environment, metadata_probe


def inspect(root, output, *, nvidia_smi, environment, expected_uuid, expected_pci, expected_name):
    """Run two bounded read-only queries and retain private metadata evidence."""
    root, output, nvidia_smi = map(Path, (root, output, nvidia_smi))
    require(all(p.is_absolute() for p in (root, output, nvidia_smi))
            and root.resolve(strict=True) == root and output.parent.resolve(strict=True) == output.parent
            and nvidia_smi.resolve(strict=True) == nvidia_smi and not os.path.lexists(output)
            and os.access(nvidia_smi, os.X_OK), 'canonical utility and fresh output required')
    require(isinstance(expected_uuid, str)
            and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', expected_uuid),
            'explicit expected GPU UUID required')
    env = clean_environment({'observer_env': environment})
    utility = file_identity(nvidia_smi)
    output.mkdir(mode=0o700)
    identity_text, identity_receipt = metadata_probe([str(nvidia_smi),
        '--query-gpu=uuid,pci.bus_id,name', '--format=csv,noheader'], root, env, output, 'identity')
    gpu = parse_device(identity_text, expected_pci=expected_pci, expected_name=expected_name)
    require(gpu['uuid'].lower() == expected_uuid.lower(), 'GPU UUID differs from reviewed declaration')
    clocks_text, clocks_receipt = metadata_probe([str(nvidia_smi), '-i', gpu['uuid'],
        '--query-supported-clocks=graphics', '--format=csv,noheader'], root, env, output, 'supported-graphics')
    rows = list(csv.reader(io.StringIO(clocks_text), skipinitialspace=True, strict=True))
    require(0 < len(rows) <= 4096, 'bounded nonempty supported-clock rows required')
    values = []
    for row in rows:
        require(len(row) == 1, 'invalid supported-clock row')
        match = re.fullmatch(r'([0-9]{1,5}) MHz', row[0].strip())
        require(match is not None and 0 < int(match.group(1)) <= 10000, 'invalid supported graphics clock')
        values.append(int(match.group(1)))
    require(1710 in values, 'requested graphics clock absent from reported application-clock list')
    current = file_identity(nvidia_smi, utility['sha256'], utility['bytes'])
    require((current['device'], current['inode']) == (utility['device'], utility['inode']),
            'NVIDIA utility replaced during metadata queries')
    return {'schema_version':1, 'diagnostic_only':True, 'gpu':gpu, 'utility':utility,
            'supported_graphics_row_count':len(values), 'requested_graphics_mhz':1710,
            'requested_graphics_in_application_list':True,
            'identity_receipt':identity_receipt, 'supported_clocks_receipt':clocks_receipt,
            'gpu_lock_support_verified':False, 'gpu_lock_permission_verified':False,
            'clock_enforcement_verified':False, 'hardware_policy_verified':False,
            'availability_confirmed':False, 'launch_ready':False,
            'scope':'Two bounded read-only NVIDIA utility queries only. The supported list is '
                    'application-clock data, not proof of GPU-lock support, permission, enforcement, '
                    'restoration, quietness, process identity or launch readiness.'}
