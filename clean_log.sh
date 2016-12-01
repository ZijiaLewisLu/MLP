#! /bin/bash

function clean {
    obj=$1
    if [ -d $obj ]; then
        ckfolder="$obj/ckpts"
        ckpt_mk="$ckfolder/checkpoint"
        flags="$obj/Flags.js"
        if ! [ -e $flags ]; then
            return 0
        fi

        if ! [ -e $ckfolder ] || ! [ -e $ckpt_mk ]; then
            echo "Remove $obj"
            rm -r $obj
        fi
    fi
    return 0
}

root=${1%/}
files=$(ls $root)
for f in $files; do
    clean "$root/$f"
done

echo
ls $root
