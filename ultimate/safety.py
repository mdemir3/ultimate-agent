"""Workspace file controls; this is not an operating-system sandbox."""
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path

SKIP = {'.git', '.ultimate', '.ssh', '.aws', '.codex', '.agents', 'node_modules', '.venv', 'venv', '__pycache__', 'dist', 'build'}
EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.md', '.txt', '.html', '.css', '.scss', '.yaml', '.yml', '.toml', '.rs', '.go', '.java', '.c', '.h', '.cpp', '.sql', '.sh'}
SECRET = re.compile(r'(-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{16,}|\bAKIA[A-Z0-9]{16}\b|(?i:api[_-]?key|password|secret|access[_-]?token)\s*[=:]\s*[\"\']?[^\s\"\']{8,})')

def ensure_no_secrets(text):
    if SECRET.search(text):
        raise ValueError('Possible credential detected; remove it before sending context to a provider.')

def digest(data):
    return hashlib.sha256(data).hexdigest()

class Workspace:
    def __init__(self, root, writable=False, check=None):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError('Workspace must be a directory.')
        self.writable = writable
        self.check = check
        self.changed = set()
        self.last_check = None
        self.snapshots = {}
        self.recovery_dir = None

    def path(self, name):
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('Use a relative path inside the workspace.')
        for part in rel.parts:
            if part in SKIP or part.startswith('.') or any(s in part.lower() for s in ('credential', 'secret', '.env')):
                raise ValueError('Protected path.')
        p = self.root / rel
        current = self.root
        for part in rel.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('Symlinks are excluded.')
        if not p.resolve().is_relative_to(self.root):
            raise ValueError('Path leaves workspace.')
        if p.name == 'ultimate.config.json':
            raise ValueError('Agent configuration is protected.')
        if p.suffix.lower() not in EXTENSIONS:
            raise ValueError('File type is outside the source-file allowlist.')
        if p.exists() and p.stat().st_nlink > 1:
            raise ValueError('Hard-linked files are excluded.')
        return p

    def listing(self):
        result = []
        for parent, dirs, names in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.') and not (Path(parent) / d).is_symlink())
            for name in sorted(names):
                rel = str((Path(parent) / name).relative_to(self.root))
                try:
                    self.path(rel)
                    result.append(rel)
                except ValueError:
                    continue
                if len(result) >= 500:
                    return {'files': result, 'truncated': True}
        return {'files': result, 'truncated': False}

    def read(self, name):
        p = self.path(name)
        if p.stat().st_size > 48000:
            raise ValueError('File exceeds the 48 KB context limit.')
        data = p.read_bytes()
        value = data.decode('utf-8')
        ensure_no_secrets(value)
        return {'path': name, 'content': value, 'sha256': digest(data)}

    def edit(self, name, content, expected_sha256):
        if not self.writable:
            raise ValueError('Writes are disabled; restart with --allow-write to authorize edits.')
        p = self.path(name)
        if len(content.encode()) > 48000:
            raise ValueError('Edit exceeds 48 KB.')
        ensure_no_secrets(content)
        old = p.read_bytes() if p.exists() else None
        actual = digest(old) if old is not None else 'NEW'
        if expected_sha256 != actual:
            # Say what is wrong without revealing the current hash, so reading stays required.
            if actual == 'NEW':
                raise ValueError('File does not exist; use NEW as expected_sha256 to create it.')
            if expected_sha256 == 'NEW':
                raise ValueError('File exists; set expected_sha256 to the sha256 from your latest read_file of it, not NEW.')
            raise ValueError('File changed or was not read. Read it again before editing.')
        p.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots.setdefault(name, old)
        if self.recovery_dir:
            originals = self.recovery_dir / 'originals'
            originals.mkdir(mode=0o700, exist_ok=True)
            manifest = {}
            for original_name, data in self.snapshots.items():
                key = str(len(manifest))
                manifest[original_name] = None if data is None else key
                if data is not None:
                    (originals / key).write_bytes(data)
            (self.recovery_dir / 'recovery.json').write_text(json.dumps(manifest, indent=2))
        with tempfile.NamedTemporaryFile(dir=p.parent, delete=False) as f:
            tmp = Path(f.name)
            f.write(content.encode())
        if p.exists():
            os.chmod(tmp, p.stat().st_mode & 0o777)
        os.replace(tmp, p)
        self.changed.add(name)
        self.last_check = None
        return {'written': name, 'sha256': digest(content.encode())}

    def verify(self):
        if not self.check:
            raise ValueError('No verification command was authorized via --check.')
        # The exact command is user-supplied. The model cannot change arguments.
        # Project tests are executable code and require a trusted workspace.
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'TMPDIR', 'LANG', 'SYSTEMROOT')}
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(self.check, cwd=self.root, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            timed_out = False
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            output.seek(0)
            raw = output.read(16000).decode('utf-8', errors='replace')
        try:
            ensure_no_secrets(raw)
        except ValueError:
            raw = '[Output withheld because a possible credential was detected.]'
        environment_error = any(marker in raw for marker in ('ModuleNotFoundError:', 'command not found', 'No module named', 'Cannot find module'))
        result = {'passed': process.returncode == 0 and not timed_out, 'environment_error': environment_error,
                  'exit_code': process.returncode, 'timed_out': timed_out, 'output': raw}
        self.last_check = result
        return result
