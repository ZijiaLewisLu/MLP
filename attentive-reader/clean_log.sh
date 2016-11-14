#! /bin/bash

root=${1%/}
files=$(ls $root)
echo $files
for f in $files; do
    f="$root/$f"
    if [ -d $f ]; then
        ckfolder="$f/ckpts"
        if [ ! -d $ckfolder ]; then
            echo "Remove $f"
            rm -r "$f"
        fi
    fi
done

ls $root
