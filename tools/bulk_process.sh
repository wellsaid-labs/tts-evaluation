#!/bin/bash -ex

PREFIX=hilary_n.01
PORT=8765

prepare()
{
    i=$1
    python3 ./prepare_script.py input/${PREFIX}.${i}.csv > input/${PREFIX}.${i}.txt
}

gentle()
{
    i=$1
    python3 ./dispatch_script_to_gentle.py -w input/${PREFIX}.${i}.wav -t input/${PREFIX}.${i}.txt --port=${PORT}
}

sample()
{
    i=$1
    python3 ./create_wav_samples_from_gentle.py input/${PREFIX}.${i}.wav input/${PREFIX}.${i}.csv input/${PREFIX}.${i}.txt.json tmp/${PREFIX}.${i}. "{\"session\": 1, \"script\": ${i}, \"talent\": \"hilary.n\"}"
}

full_set()
{
    entry_set=$1
    for entry in $entry_set; do
        prepare $entry
    done
    for entry in $entry_set; do
        gentle $entry
    done
    for entry in $entry_set; do
        sample $entry
    done
}

range_set()
{
    from=$1
    to=$2
    full_set "$(seq $from $to)"
}

# range_set 1 15
full_set "16-21 22-27 28-33 34-39"
