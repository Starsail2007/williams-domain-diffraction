#!/bin/zsh
set -eu
TASK_ARGUMENT_COUNT=$#
TASK_DOWNLOAD=''
function task_cleanup {
  local TASK_EXIT=$?
  if [[ -n "$TASK_DOWNLOAD" && -f "$TASK_DOWNLOAD" ]]; then rm -f "$TASK_DOWNLOAD"; fi
  if (( TASK_EXIT != 0 )); then
    print -u2 '启动没有完成。请保留上方错误；首次准备环境时请检查网络及文件夹写入权限，再次启动会重试。'
    if [[ -t 0 && $TASK_ARGUMENT_COUNT -eq 0 ]]; then read -r '?按回车关闭'; fi
  fi
  return "$TASK_EXIT"
}
trap task_cleanup EXIT
TASK_ROOT=${0:A:h}
cd "$TASK_ROOT"
case "$(uname -m)" in
  arm64) TASK_TARGET=macos-arm64 ;;
  x86_64) TASK_TARGET=macos-x86_64 ;;
  *) print '此包支持Apple Silicon及Intel Mac。'; exit 1 ;;
esac
TASK_MAJOR=$(sw_vers -productVersion)
if (( ${TASK_MAJOR%%.*} < 14 )); then
  print '本便携环境需要macOS 14或更新版本；阅读START_HERE.html不受此限制。'
  exit 1
fi
TASK_PYTHON="$TASK_ROOT/.runtime/$TASK_TARGET/python/bin/python3"
TASK_LOCK="$TASK_ROOT/configs/environment_lock.json"
TASK_SHA=$(/usr/bin/plutil -extract "platforms.$TASK_TARGET.runtime.sha256" raw -o - "$TASK_LOCK")
TASK_ORIGIN="$TASK_ROOT/.runtime/$TASK_TARGET/runtime_archive.sha256"
TASK_INSTALLED_SHA=''
if [[ -f "$TASK_ORIGIN" ]]; then TASK_INSTALLED_SHA=$(<"$TASK_ORIGIN"); fi
if [[ ! -x "$TASK_PYTHON" || "$TASK_INSTALLED_SHA" != "$TASK_SHA" ]]; then
  print '首次启动：正在从官方源准备项目内Python环境，需要联网……'
  TASK_LOCK="$TASK_ROOT/configs/environment_lock.json"
  TASK_URL=$(/usr/bin/plutil -extract "platforms.$TASK_TARGET.runtime.url" raw -o - "$TASK_LOCK")
  TASK_SHA=$(/usr/bin/plutil -extract "platforms.$TASK_TARGET.runtime.sha256" raw -o - "$TASK_LOCK")
  TASK_FILENAME=$(/usr/bin/plutil -extract "platforms.$TASK_TARGET.runtime.filename" raw -o - "$TASK_LOCK")
  mkdir -p "$TASK_ROOT/.runtime/downloads"
  TASK_ARCHIVE="$TASK_ROOT/.runtime/downloads/$TASK_FILENAME"
  if [[ ! -f "$TASK_ARCHIVE" ]] || [[ $(shasum -a 256 "$TASK_ARCHIVE" | cut -d ' ' -f 1) != "$TASK_SHA" ]]; then
    TASK_DOWNLOAD="$TASK_ARCHIVE.part.$$"
    curl --fail --location --retry 2 --connect-timeout 30 "$TASK_URL" --output "$TASK_DOWNLOAD"
    printf '%s  %s\n' "$TASK_SHA" "$TASK_DOWNLOAD" | shasum -a 256 -c -
    mv "$TASK_DOWNLOAD" "$TASK_ARCHIVE"
  fi
  TASK_STAGING=$(mktemp -d "$TASK_ROOT/.runtime/$TASK_TARGET.unpack.XXXXXX")
  tar -xzf "$TASK_ARCHIVE" -C "$TASK_STAGING"
  if [[ -e "$TASK_ROOT/.runtime/$TASK_TARGET" ]]; then
    mv "$TASK_ROOT/.runtime/$TASK_TARGET" "$TASK_ROOT/.runtime/$TASK_TARGET.incomplete.$(date +%Y%m%d%H%M%S).$$"
  fi
  mv "$TASK_STAGING" "$TASK_ROOT/.runtime/$TASK_TARGET"
  printf '%s\n' "$TASK_SHA" > "$TASK_ORIGIN"
fi
"$TASK_PYTHON" -I -B -X utf8 "$TASK_ROOT/scripts/bootstrap.py" --platform "$TASK_TARGET"
TASK_STATUS=0
"$TASK_PYTHON" -I -B -X utf8 "$TASK_ROOT/scripts/runtime.py" "$@" || TASK_STATUS=$?
exit "$TASK_STATUS"
