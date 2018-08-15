# A small collection of tools

## Dispatch a script and a wav file to gentle for parsing

  * Start gentle using either the prepackaged dmg or building from source

  * Prepare the script CSV file for gentle:

```
    $ python3 ./prepare_script.py script1.csv > script1.txt
```
  * Dispatch the script to gentle:

```
    $ python3 ./dispatch_script_to_gentle.py -w script1.wav -t script1.txt --port=63916
```

This will generate a script1.txt.json file, containing the response from gentle.

  * Slice the wav file up, using the gentle response file:

```
    $ python3 ./create_wav_samples_from_gentle.py
    usage: create_wav_samples_from_gentle.py [-h] [-n] wav script gentle dest tags
    create_wav_samples_from_gentle.py: error: the following arguments are required: wav, script, gentle, dest, tags

    $ mkdir tmp
    $ python3 create_wav_samples_from_gentle.py script1.wav script1.csv script1.txt.json tmp/script1_ \"{\"script\": 1, \"talent\": \"hilary\"}
```

Note: it is possible to pass in arbitrary JSON as the last parameter; this will be populated along with script and timing information into a JSON file next to the resulting wav file.

Note: the destination directory - `tmp/` in the example above - must be created prior to this command.
