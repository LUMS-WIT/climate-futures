#!/usr/bin/env bash
# Rebuild the student worksheet PDFs in content-src/activities/ from the HTML
# sources in this folder, using headless Chrome. Run locally after editing a
# source file, then commit the regenerated PDF. CI does not run this.
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
for f in *.html; do
  out="../${f%.html}.pdf"
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$out" "file://$PWD/$f" 2>/dev/null
  echo "built $out"
done
