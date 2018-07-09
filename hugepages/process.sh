#!/bin/bash
RESULTFILE="$1"
TMPOUT=$(mktemp)

grep -E '(seconds|slots|instructions|cache-misses)' "$RESULTFILE" | awk '{if (NR%4==0) printf ",%f", $1; else if (NR%4==1) printf "%d", $8; else printf ",%f", $4; if (NR%4==0) printf "\n"}' > "$TMPOUT"

(printf 'file="%s"\n' "$TMPOUT"; cat plot.r) | R --vanilla >/dev/null
mv Rplots.pdf "$RESULTFILE.pdf"

rm "$TMPOUT"
