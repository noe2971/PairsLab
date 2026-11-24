#!/usr/bin/env bash
# Installs the commit-msg hook that strips AI-attribution lines from commit messages.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
HOOK="$ROOT/.git/hooks/commit-msg"
cat > "$HOOK" <<'HOOK_EOF'
#!/usr/bin/env bash
# Delete lines starting with "Co-Authored-By:" (any case) or containing "Generated with".
FILE="$1"
grep -viE '^co-authored-by:|generated with' "$FILE" > "$FILE.tmp" || true
mv "$FILE.tmp" "$FILE"
HOOK_EOF
chmod +x "$HOOK"
echo "Installed commit-msg hook at $HOOK"
