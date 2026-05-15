#!/bin/sh
# commit-msg hook: strip the Cursor co-author trailer injected by the IDE.
# Install: cp scripts/strip-cursor-trailer.sh .git/hooks/commit-msg && chmod +x .git/hooks/commit-msg
sed -i '' "/^Co-authored-by: Cursor <cursoragent@cursor\.com>$/d" "$1" 2>/dev/null \
  || sed -i    "/^Co-authored-by: Cursor <cursoragent@cursor\.com>$/d" "$1"
