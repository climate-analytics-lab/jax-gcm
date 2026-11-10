"""Bounded background throughput screens on 8- and 32-vCPU C4D Spot VMs."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time

HERE = Path(__file__).resolve().parent
MACHINES = ("c4d-standard-8", "c4d-standard-32")
ZONES = ("us-central1-a", "us-central1-b", "us-central1-c")


def write_json(path, data):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def screen(shape, root, config, archive, sha):
    directory = root / shape
    directory.mkdir(exist_ok=True)
    state = {'machine_type': shape, 'phase': 'starting', 'started_at': datetime.now(timezone.utc).isoformat(),
             'simulated_days': 1 / 24, 'timed_repeats': 3, 'attempts': [],
             'scope': 'one-hour throughput screen; no full-month run or checkpoint recovery'}

    def save():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(directory / 'status.json', state)

    save()
    try:
        with (directory / 'supervisor.log').open('a', buffering=1) as log:
            for zone in ZONES:
                state['phase'] = 'requesting_capacity'
                attempt = {'zone': zone, 'requested_at': datetime.now(timezone.utc).isoformat()}
                state['attempts'].append(attempt)
                save()
                command = [sys.executable, '-I', '-u', str(HERE / 'launch.py'),
                           '--config', str(config), '--source-archive', str(archive),
                           '--source-sha256', sha, '--machine-type', shape, '--zone', zone,
                           '--total-days', str(1 / 24), '--save-days', str(1 / 24),
                           '--repeats', '3', '--truncation', '63', '--output', str(directory / 'results')]
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                output = []
                for line in process.stdout:
                    log.write(line)
                    output.append(line)
                    if line.startswith('Results: '):
                        attempt['gcs_output'] = line.removeprefix('Results: ').strip()
                        attempt['run_id'] = attempt['gcs_output'].split('/')[4]
                        save()
                    if line.startswith('Running echam-spot-'):
                        attempt['instance'] = line.split()[1].rstrip(';')
                        attempt['vm_ready_at'] = datetime.now(timezone.utc).isoformat()
                        state['phase'] = 'running'
                        save()
                attempt['exit_code'] = process.wait()
                text = ''.join(output)
                if process.returncode == 0:
                    artifacts = directory / 'results' / attempt['run_id'] / 'artifacts'
                    artifacts.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(artifacts.parent / 'results.tar.gz') as bundle:
                        bundle.extractall(artifacts, filter='data')
                    result = json.loads((artifacts / 'results.json').read_text())
                    validation = json.loads((artifacts / 'validation.json').read_text())
                    if result['platform'] != 'cpu' or not validation['finite']:
                        raise RuntimeError('CPU backend or finite-output validation failed')
                    state.update(phase='complete', summary=result, validation=validation,
                                 artifacts=str(artifacts), vm_cleanup_confirmed='Deleted echam-spot-' in text,
                                 projected_vm_hours_per_365day_year=result['mean_s'] / (1 / 24) * 365 / 3600)
                    if (artifacts / 'cpu-resources.json').exists():
                        state['resources'] = json.loads((artifacts / 'cpu-resources.json').read_text())
                    save()
                    return
                if 'ZONE_RESOURCE_POOL_EXHAUSTED' not in text:
                    state.update(phase='failed', error=text[-6000:])
                    save()
                    return
                attempt['error'] = 'capacity_stockout'
                save()
            state['phase'] = 'capacity_unavailable'
            save()
    except Exception as error:
        state.update(phase='failed', error=str(error))
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=HERE.parents[1])
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    batch = {'phase': 'waiting_for_authentication', 'machines': MACHINES,
             'started_at': datetime.now(timezone.utc).isoformat(), 'authentication_wait_hours': 24}
    write_json(root / 'batch.json', batch)
    try:
        deadline = time.monotonic() + 24 * 3600
        while time.monotonic() < deadline:
            auth = subprocess.run(['gcloud', 'auth', 'print-access-token'], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=60)
            if auth.returncode == 0:
                break
            time.sleep(60)
        else:
            raise RuntimeError('Authentication wait expired; no VMs created')
        config = json.loads(args.config.read_text())
        config['image'] = subprocess.check_output(['gcloud', 'compute', 'images', 'describe-from-family',
            'ubuntu-2404-lts-amd64', '--project=ubuntu-os-cloud', '--format=value(selfLink)'], text=True, timeout=60).strip()
        config_path = root / 'config.json'
        write_json(config_path, config)
        spec = importlib.util.spec_from_file_location('spot_launch', HERE / 'launch.py')
        launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launch)
        archive = root / 'source.tar.gz'
        sha, info = launch.snapshot(args.source.resolve(), archive)
        write_json(root / 'source-manifest.json', {'sha256': sha, **info})
        batch.update(phase='running', source_sha256=sha, image=config['image'])
        write_json(root / 'batch.json', batch)
        workers = [threading.Thread(target=screen, args=(shape, root, config_path, archive, sha)) for shape in MACHINES]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        phases = {shape: json.loads((root / shape / 'status.json').read_text())['phase'] for shape in MACHINES}
        batch.update(phase='complete' if all(value == 'complete' for value in phases.values()) else 'finished_with_errors',
                     jobs=phases, finished_at=datetime.now(timezone.utc).isoformat())
        write_json(root / 'batch.json', batch)
    except Exception as error:
        batch.update(phase='failed', error=str(error))
        write_json(root / 'batch.json', batch)
        raise


if __name__ == '__main__':
    main()
