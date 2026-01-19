#!/bin/sh

BASE="slurm/out"
current=""

while true; do
    newest=$(find "$BASE" -type f -printf '%T@ %p\n' 2>/dev/null \
              | sort -nr | head -n1 | cut -d' ' -f2-)

    if [ "$newest" != "$current" ] && [ -n "$newest" ]; then
        echo "Watching: $newest"
        pkill -P $$ tail 2>/dev/null
        tail -F "$newest" &
        current="$newest"
    fi

    sleep 2
done
