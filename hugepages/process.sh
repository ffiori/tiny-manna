#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <optimized.txt> <original.txt>"
    exit 1
fi

RESULTFILE1="$1"
RESULTFILE2="$2"
TMPOUT1=$(mktemp)
TMPOUT2=$(mktemp)

grep -E '(seconds|slots|instructions|cache-misses)' "$RESULTFILE1" | awk '{if (NR%4==0) printf ",%f", $1; else if (NR%4==1) printf "%d", $8; else printf ",%f", $4; if (NR%4==0) printf "\n"}' > "$TMPOUT1"
grep -E '(seconds|slots|instructions|cache-misses)' "$RESULTFILE2" | awk '{if (NR%4==0) printf ",%f", $1; else if (NR%4==1) printf "%d", $8; else printf ",%f", $4; if (NR%4==0) printf "\n"}' > "$TMPOUT2"

(printf 'filefast="%s"\nfileslow="%s"\n' "$TMPOUT1" "$TMPOUT2"; cat plot.r) | R --vanilla >/dev/null
mv Rplots.pdf "$RESULTFILE1+$RESULTFILE2.pdf"

rm "$TMPOUT1" "$TMPOUT2"
