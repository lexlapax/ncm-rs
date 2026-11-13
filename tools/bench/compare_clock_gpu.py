"""Compare supplied read-only GPU metadata records; never invoke a GPU command."""
from copy import deepcopy
from pathlib import Path
import re

from clock_library_contract import signature
from compare_clock_snapshots import validate as validate_snapshot
from readonly_trace_preflight import require
from validate_readonly_lifecycle import receipt_complete

FALSE_CLAIMS = ('gpu_lock_support_verified','gpu_lock_permission_verified','clock_enforcement_verified',
                'hardware_policy_verified','availability_confirmed','launch_ready')


def validate(record, *, root):
    require(isinstance(record,dict) and type(record.get('schema_version')) is int and record['schema_version']==1
            and record.get('diagnostic_only') is True and all(record.get(k) is False for k in FALSE_CLAIMS),
            'read-only GPU metadata record required')
    root=Path(root)
    require(root.is_absolute() and '..' not in root.parts, 'absolute GPU query cwd required')
    utility=record['utility']; signature(utility)
    gpu=record['gpu']; uuid=gpu['uuid']
    require(isinstance(uuid,str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',uuid)
            and isinstance(gpu['pci_bus_id'],str) and re.fullmatch(r'[0-9A-F]{8}:[0-9A-F]{2}:[0-9A-F]{2}\.[0-7]',gpu['pci_bus_id'])
            and isinstance(gpu['name'],str) and bool(gpu['name'])
            and type(gpu['visible_device_count']) is int and gpu['visible_device_count']==1
            and gpu['historical_uuid_equality_verified'] is False, 'invalid GPU identity declaration')
    require(type(record['supported_graphics_row_count']) is int and 0 < record['supported_graphics_row_count'] <= 4096
            and type(record['requested_graphics_mhz']) is int and record['requested_graphics_mhz']==1710
            and record['requested_graphics_in_application_list'] is True, 'invalid supported-clock declaration')
    commands=(('identity_receipt',[utility['path'],'--query-gpu=uuid,pci.bus_id,name','--format=csv,noheader']),
              ('supported_clocks_receipt',[utility['path'],'-i',uuid,'--query-supported-clocks=graphics','--format=csv,noheader']))
    for name,argv in commands:
        receipt=record[name]; receipt_complete(receipt)
        require(receipt['argv']==argv and receipt['cwd']==str(root)
                and receipt['timeout_seconds']==10 and receipt['output_limit_bytes']==1024**2
                and receipt['termination_grace_seconds']==0 and receipt['report_limit_bytes'] is None
                and type(receipt['captured_bytes']) is int and 0 < receipt['captured_bytes'] <= 1024**2
                and receipt['captured_bytes']==receipt['observed_bytes']
                and isinstance(receipt['capture_sha256'],str)
                and re.fullmatch('[0-9a-f]{64}',receipt['capture_sha256']), 'GPU metadata receipt differs')
    return record


def compare(reviewed, current, *, root):
    """Compare stable identity/utility/output digests; supplied records only."""
    reviewed,current=(validate(deepcopy(value),root=root) for value in (reviewed,current))
    for key in ('gpu','utility','supported_graphics_row_count','requested_graphics_mhz','requested_graphics_in_application_list'):
        require(reviewed[key]==current[key], 'GPU metadata differs: '+key)
    for key in ('identity_receipt','supported_clocks_receipt'):
        for field in ('captured_bytes','capture_sha256'):
            require(reviewed[key][field]==current[key][field], 'GPU query output differs: '+key+'.'+field)
    return {'schema_version':1,'supplied_gpu_metadata_matches':True,'fresh_producer_authenticated':False,
            'gpu_lock_permission_verified':False,'hardware_policy_verified':False,'launch_ready':False,
            'scope':'Supplied read-only metadata and capture digests only; no current-file reads, '
                    'producer authentication, lock permission/enforcement, restoration or launch authorization.'}


def bind_snapshot(snapshot, gpu_record):
    """Bind supplied GPU metadata to the exact utility in a supplied file snapshot.

    This does not authenticate either producer, independently read utility bytes,
    or establish that the device is currently available or safe to modify.
    """
    snapshot = validate_snapshot(deepcopy(snapshot))
    gpu_record = validate(deepcopy(gpu_record), root=snapshot['project_root'])
    utilities = snapshot['utilities']
    require(isinstance(utilities, list) and len(utilities) == 3
            and signature(utilities[1]) == signature(gpu_record['utility']),
            'GPU metadata utility differs from file snapshot')
    matches = [value for value in snapshot['inventory']['frozen_identities']
               if value['path'] == gpu_record['utility']['path']]
    require(len(matches) == 1 and signature(matches[0]) == signature(gpu_record['utility']),
            'GPU utility is not bound in frozen recheck inventory')
    return {'schema_version':1,'supplied_gpu_utility_binding_matches':True,
            'gpu_uuid':gpu_record['gpu']['uuid'], 'producer_authentication_verified':False,
            'gpu_lock_permission_verified':False,'launch_ready':False,
            'scope':'Consistency between supplied file and GPU metadata records only; '
                    'not producer authentication, current device state or launch authorization.'}
