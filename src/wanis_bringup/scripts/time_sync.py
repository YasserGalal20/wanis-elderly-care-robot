#!/usr/bin/env python3
"""
time_sync.py — Push the server's wall clock to the Raspberry Pi over SSH with
NTP-style RTT compensation.

HOW IT WORKS
  1. Open one persistent paramiko SSH connection (avoids per-command TCP
     handshake that inflates RTT to 100-200 ms).
  2. Take N lightweight probes to measure the SSH round-trip time.
  3. Use the probe with minimum RTT (least network jitter) for the offset
     calculation — same heuristic used by NTP.
  4. When issuing 'sudo date -s @<ts>', compensate target by +RTT/2 so the
     timestamp the Pi receives is as close to server_now as possible.
  5. Optionally trigger 'sudo chronyc settime <ts>' first (softer step if
     chrony is running on the Pi) and fall back to 'sudo date -s'.
  6. Verify with a second round of probes.

WHY NOT ntpdate?
  'ntpdate pool.ntp.org' syncs the Pi to an internet NTP pool — not to this
  server. For a robot on a closed LAN (possibly without internet), the server
  IS the reference clock. We push server→Pi, not pull from the net.

BEST PERMANENT SOLUTION (chrony LAN setup, one-time, ~1 ms accuracy):
  On the SERVER:
      sudo bash -c '
        grep -q "local stratum" /etc/chrony.conf || echo "local stratum 8" >> /etc/chrony.conf
        grep -q "^allow" /etc/chrony.conf      || echo "allow 192.168.0.0/16" >> /etc/chrony.conf
        systemctl enable --now chrony
        systemctl restart chrony'

  On the PI (replace SERVER_IP):
      sudo bash -c '
        sed -i "/^pool\|^server/d" /etc/chrony.conf
        echo "server SERVER_IP iburst prefer" >> /etc/chrony.conf
        systemctl enable --now chrony
        systemctl restart chrony
        chronyc -a makestep'

  After this, no script is needed — the Pi stays in sync automatically.
  Run this script only as a fallback for sessions before the daemon is set up.

SUDOERS on Pi (required for sudo date -s and sudo chronyc):
  echo "USER ALL=(root) NOPASSWD: /bin/date, /usr/bin/chronyc" | \\
    sudo tee /etc/sudoers.d/time_sync
  sudo chmod 440 /etc/sudoers.d/time_sync
  (replace USER with the SSH login name)
"""

import os
import argparse
import sys
import time

import paramiko

DEFAULT_HOST     = "192.168.50.82"
DEFAULT_USER     = "yasser"
DEFAULT_PASSWORD = os.environ.get("ROBOT_SSH_PASSWORD", "")
DEFAULT_PORT     = 22
RTT_SAMPLES      = 7       # more samples → better minimum-RTT estimate
SSH_TIMEOUT      = 10      # seconds for initial connection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_ssh(host, user, port, password, key_path) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = dict(hostname=host, port=port, username=user, timeout=SSH_TIMEOUT)
    if key_path:
        kwargs["key_filename"] = key_path
    elif password:
        kwargs["password"] = password
    client.connect(**kwargs)
    return client


def _exec(client: paramiko.SSHClient, cmd: str, timeout: int = 8) -> tuple[int, str, str]:
    """Run a command; return (exit_code, stdout, stderr)."""
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    rc  = stdout.channel.recv_exit_status()
    return rc, out, err


def _probe_once(client: paramiko.SSHClient) -> tuple[float, float]:
    """Single probe: returns (rtt_seconds, pi_timestamp_unix)."""
    t0 = time.time()
    rc, out, _ = _exec(client, "date +%s.%N")
    t1 = time.time()
    if rc != 0 or not out:
        raise RuntimeError(f"date probe failed (rc={rc})")
    return t1 - t0, float(out)


def _best_probe(client: paramiko.SSHClient, n: int) -> tuple[float, float, float]:
    """Take N probes; return (best_rtt, pi_ts, server_mid) for minimum-RTT sample."""
    best_rtt  = float("inf")
    best_pi   = 0.0
    best_smid = 0.0
    for i in range(n):
        t0 = time.time()
        rtt, pi_ts = _probe_once(client)
        smid = t0 + rtt / 2.0
        if rtt < best_rtt:
            best_rtt, best_pi, best_smid = rtt, pi_ts, smid
        # small pause to let network jitter vary between samples
        time.sleep(0.05)
    return best_rtt, best_pi, best_smid


# ---------------------------------------------------------------------------
# Clock-set strategies
# ---------------------------------------------------------------------------

def _try_chronyc_settime(client: paramiko.SSHClient, target_ts: float) -> bool:
    """Ask chrony to step the clock.  Returns True on success."""
    # chronyc settime expects 'YYYY-MM-DD HH:MM:SS' in UTC
    import datetime
    dt = datetime.datetime.utcfromtimestamp(target_ts)
    ts_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    rc, out, err = _exec(client, f"sudo chronyc settime '{ts_str}'")
    if rc == 0:
        print(f"  chronyc settime → OK  ({out})")
        # Tell chrony to apply the step immediately
        _exec(client, "sudo chronyc makestep")
        return True
    print(f"  chronyc settime skipped (rc={rc}: {err or out})")
    return False


def _set_date(client: paramiko.SSHClient, target_ts: float) -> bool:
    """Set the Pi's clock via 'sudo date -s @<ts>'.  Returns True on success."""
    rc, out, err = _exec(client, f"sudo -n /bin/date -u -s @{target_ts:.6f}")
    if rc == 0:
        print(f"  sudo date -s → OK  ({out})")
        return True
    print(
        f"  sudo date -s FAILED (rc={rc}: {err or out})\n"
        "  Ensure the SSH user can run `sudo -n /bin/date` on the Pi.\n"
        "  Add to /etc/sudoers.d/time_sync:\n"
        f"      {DEFAULT_USER} ALL=(root) NOPASSWD: /bin/date, /usr/bin/chronyc",
        file=sys.stderr,
    )
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Push server clock to Raspberry Pi over SSH (NTP-style RTT compensation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host",     default=DEFAULT_HOST,     help=f"Pi IP/hostname (default: {DEFAULT_HOST})")
    parser.add_argument("--user",     default=DEFAULT_USER,     help=f"SSH username (default: {DEFAULT_USER})")
    parser.add_argument("--port",     type=int, default=DEFAULT_PORT, help=f"SSH port (default: {DEFAULT_PORT})")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="SSH password (plain-text; prefer key auth)")
    parser.add_argument("--identity", "-i",                     help="SSH private key path")
    parser.add_argument("--samples",  type=int, default=RTT_SAMPLES,
                        help=f"RTT probe count for minimum-RTT estimate (default: {RTT_SAMPLES})")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Skip update if |offset| < threshold seconds (default: 0 = always update)")
    args = parser.parse_args()

    # ── Connect ─────────────────────────────────────────────────────────────
    print(f"[time_sync] Connecting to {args.user}@{args.host}:{args.port} ...")
    try:
        client = _open_ssh(args.host, args.user, args.port, args.password, args.identity)
    except Exception as e:
        print(f"[time_sync] SSH connection failed: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        # ── Phase 1: measure offset ──────────────────────────────────────────
        print(f"[time_sync] Taking {args.samples} RTT probes ...")
        try:
            best_rtt, pi_ts, server_mid = _best_probe(client, args.samples)
        except Exception as e:
            print(f"[time_sync] Probe failed: {e}", file=sys.stderr)
            sys.exit(3)

        offset = pi_ts - server_mid
        print(f"[time_sync] Best RTT:           {best_rtt * 1000:.1f} ms")
        print(f"[time_sync] Pi clock:           {pi_ts:.3f}")
        print(f"[time_sync] Server clock (mid): {server_mid:.3f}")
        # Keep "Offset (pi - server):" exact — app.py parses this line to extract ms
        print(f"[time_sync] Offset (pi - server): {offset:+.3f} s")

        if abs(offset) < args.threshold:
            print(f"[time_sync] Within threshold ({args.threshold:.3f} s) — no update needed.")
            sys.exit(0)

        # ── Phase 2: set clock ───────────────────────────────────────────────
        # Target = server_now + RTT/2 — by the time the remote command runs,
        # approximately half an RTT will have elapsed from when we call time.time().
        print("[time_sync] Updating Pi clock ...")
        target = time.time() + best_rtt / 2.0

        ok = _try_chronyc_settime(client, target)
        if not ok:
            ok = _set_date(client, target)
        if not ok:
            sys.exit(4)

        # ── Phase 3: verify ──────────────────────────────────────────────────
        time.sleep(0.1)  # let the clock settle
        print("[time_sync] Verifying ...")
        try:
            v_rtt, v_pi, v_smid = _best_probe(client, max(3, args.samples // 2))
        except Exception as e:
            print(f"[time_sync] Verify probe failed: {e}", file=sys.stderr)
            sys.exit(5)

        new_offset = v_pi - v_smid
        # app.py takes the LAST "Offset (pi - server):" line — this post-sync one is it
        print(f"[time_sync] Offset (pi - server): {new_offset:+.3f} s  (post-sync verify)")
        print(f"[time_sync] RTT during verify:    {v_rtt * 1000:.1f} ms")

        if abs(new_offset) > 0.300:
            print(
                f"[time_sync] WARNING: offset {new_offset * 1000:+.0f} ms still exceeds 300 ms — "
                "consider the permanent chrony LAN setup (see script header).",
                file=sys.stderr,
            )
            sys.exit(6)

        print("[time_sync] Done — Pi clock synchronized.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
