#!/usr/bin/env bash
set -euo pipefail
version=0.160.1
cli_archive="/tmp/codex-x86_64-unknown-linux-musl-v${version}.tar.gz"
cli_dir="/tmp/codex-cli-v${version}"
cli="/tmp/codex-v${version}"
if test ! -x "$cli"; then
  curl -fsSL "https://github.com/openai/codex/releases/download/rust-v${version}/codex-x86_64-unknown-linux-musl.tar.gz" -o "$cli_archive"
  mkdir -p "$cli_dir"
  tar -xzf "$cli_archive" -C "$cli_dir"
  install -m 755 "$cli_dir/codex-x86_64-unknown-linux-musl" "$cli"
fi
host_asset="codex-code-mode-host-x86_64-unknown-linux-musl.tar.gz"
host_archive="/tmp/$host_asset"
host_dir="/tmp/codex-code-mode-host-v${version}"
host="/tmp/codex-code-mode-host"
if test ! -x "$host"; then
  curl -fsSL "https://github.com/openai/codex/releases/download/rust-v${version}/$host_asset" -o "$host_archive"
  printf '%s  %s\n' 8a69207d97545ac753b6585974e1e67a4c51ae5deacf06517db512bb25e0e3c2 "$host_archive" | sha256sum -c -
  mkdir -p "$host_dir"
  tar -xzf "$host_archive" -C "$host_dir"
  install -m 755 "$host_dir/codex-code-mode-host-x86_64-unknown-linux-musl" "$host"
fi
test -x "$host"
"$cli" --version
