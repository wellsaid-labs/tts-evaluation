#!/bin/bash -ex

PREFIX=hilary_n.01

for i in {1..15}; do
    python3 ./prepare_script.py input/${PREFIX}.${i}.csv > input/${PREFIX}.${i}.txt
done

for i in {1..15}; do
    python3 ./dispatch_script_to_gentle.py -w input/${PREFIX}.${i}.wav -t input/${PREFIX}.${i}.txt --port=63916
done

for i in {1..15}; do
    python3 ./create_wav_samples_from_gentle.py input/${PREFIX}.${i}.wav input/${PREFIX}.${i}.csv input/${PREFIX}.${i}.txt.json tmp/${PREFIX}.${i}. "{\"session\": 1, \"script\": ${i}, \"talent\": \"hilary.n\"}"
done
