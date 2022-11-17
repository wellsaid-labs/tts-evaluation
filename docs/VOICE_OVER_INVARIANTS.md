
# Voice-Over Invariants

## Pauses & Breaks

We have decided that the longest meaningful pause is around 1 second long. If a pause is longer
than that, we consider that pause to be a break instead.

We use a couple of different methods for identifying pauses or breaks:

- We look for audio segments that are quieter than -50 db. We are sometimes more restrictive to
  increase data quality.
- We start looking 50 or more milliseconds after a speech segment. This accounts for unvoiced
  speech sounds at the end of a word, like "s".
- We look for audio segments that are not in between the syllables of a word.

With regards to recording datasets, we ask voice-actors that their silences are quieter than -60 db.
Ideally, they are more like -70 db or lower.

## Segmentation

In normal speech, one typically finds many consecutive words being said with no pauses between them,
and often the final sounds of one word blend smoothly or fuse with the initial sounds of the next
word. For this reason, we segment speech on pauses as defined above.

The smallest segment that we allow is 0.1 to 0.2 seconds, depending on the application. Google
STT also has a similar constraint at 0.1 seconds.

Keep in mind, most conventional TTS datasets segment speech based on word boundaries. Due to the
intractability of that task as many as 1 out of 10 clips will end preemptively in these datasets.
We have often referred to an audio clip ending early as "audio cutoff" or "word cutoff".

## Stops

We cut the audio such that the voice-over ends, generally, within 200 milliseconds of the last
voiced sound. The model relies on this consistency to learn when to stop generation.

## Speed

To help monitor for errors, we like to define the slowest reading speed to be around
0.2 seconds per character. The fastest reading speed that we accept is 0.04 seconds per character.

## Sessions

We recognize that each voice-over session has a distinct sound. Even within the same voice-over
session, a voice actor may sound different at the beginning and ending of the session. To help
increase consistency, we limit sessions to 15-minutes in length and give the model session context.

A voice-actor, even during a session, may not be consistent. We also have some datasets with much
longer sessions. We have some work to-do to resolve this inconsistency. We've found, sometimes,
that longer sessions indicates that the voice-actor stitched together multiple files from separate
recording sessions.

## Yes / No Questions

To help us better model question inflection, we decided that yes / no questions should have
an upwards inflection. We have created several training and test datasets to enforce that.

## Spectrograms

Following previous literature, we use spectrograms to represent audio. Spectrograms are
are well-suited for modeling human hearing. To help drive that further, we apply
several transformations to our spectrograms, they include:

- We weigh frequencies based on the human ears sensitivity to them.
- We bucket frequencies based on the human ears ability to tell them apart.
- We bound loudness based on the human ears ability to tell apart different levels
  of silence.
- We bound frequencies based on the human ears ability to hear high and low pitches.

Unfortunately, this isn't an exact science, so we are interested in finding simpler solution to
such an integral component of our model.

## Loudness

We expect that our datasets are normalized to -22 LUFS. That means, on average, with silences
included, the loudness is -22 LUFS. This helps ensure that our voices are at an industry standard
loudness.

Loudness is also difficult to define due to the peculiarities in human-hearing. Depending
on the application, we use different algorithms, including:

- RMS: This is the easiest and simplest algorithm to implement.
- ITU-R BS.1770 (LUFS): This algorithm is the industry standard.
- Custom: We calculate loudness from our spectrograms.
