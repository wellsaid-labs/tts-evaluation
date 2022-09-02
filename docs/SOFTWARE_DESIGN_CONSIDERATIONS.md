# Software Design Considerations

At a high-level, here are some of the considerations we made when designing this software. To help
keep this software consistent, please adhere to these guidelines :nail_care:

## Architecture

["Functional Core, Imperative Shell"](https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell),
is our guiding principle for architecting our code base.

In our case, `lib` is the functional core of our application and `run` is the imperative shell that
interacts with the the external world.

This structure is akin to using open-source packages. The modules in `lib` could be open sourced
because they general and self-sufficient. The modules in `run` are more specific to our needs
and more coupled together.

Learn more:

- <https://www.destroyallsoftware.com/talks/boundaries>
- <https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell>
- <https://www.javiercasas.com/articles/functional-programming-patterns-functional-core-imperative-shell>
- <https://github.com/kbilsted/Functional-core-imperative-shell/blob/master/README.md>
- <https://en.wikipedia.org/wiki/Pure_function>

### Services `run`

Within run, we categorize our modules. The modules with underscores are not runnable. For examples,
our utility functions are not runnable. On the other hand, our training scripts, that don't
have underscores, are runnable using the terminal.

For us it's particularly important to document our assumptions, we do so in the `_config` module.
This module allows us to configure hyperparameters across the board from our data processing
to training.

### Library `lib`

The functions in `lib` reliable and well-tested as they constitute a functional core that
the rest of the application relies on.

## Dataset Invariants

To model text-to-speech, we rely on our datasets to be consistent. We assume that there is a
one-to-one transformation from text to speech that the model can learn. Our hope is to create
a text-to-speech that is akin to a piano, it's intuitive to use and beautiful to listen to.

With that in mind, we have defined a couple of invariants that we rely on...

(Keep in mind that many of these invariants were created for English)

### Language Invariants

While building this software, we have thought about language as a tool for creating voice-over.
In doing so, we have restricted some of the ways we interpret written text.

#### Progression

We have restricted our model to read left-to-right without skipping around the text. To support
this, we have also normalized our scripts to verbalize phrases like "$5 million" which is
read "five million dollars" instead of "dollar five million".

With that in mind, we expect that the voice-over stops after the last letter is read by the model.

#### Letter Casing

We have decided, in order to reduce ambiguity, that capital letters should only be used to represent
initialisms.

There are a couple ways that this has been implemented. For example, we check if any capital letters
within a script correspond to capital letters in the transcript. Also, we consider acronyms to be
non-standard words that are verbalized into standard words (i.e. "NCAA" to "NC double A"). We
have also created datasets that have only initialisms, and no acronyms.

#### Alphabet

We have decided to restrict the alphabet to only ASCII characters, with a few exceptions.
The terminology that we like to use internally is "readable characters". We hope that
the alphabet closely aligns with the characters voice-actors can read. This also includes
normalizing some rarely used characters like backticks, curly braces, etc.

#### Non-Standard Words

We verbalize or remove non-standard words from our training data, this includes slashes,
numbers, ampersands, etc.

Slashes (i.e. "us lash") have a particularly notorious history at WellSaid Labs. We didn't
realize that slashes can both be voiced or silent depending on the context. All in all, our
model learned to stay silent, even when our users wrote out the word "slash".

### Voice-Over Invariants

#### Pauses & Breaks

We have decided that the longest meaningful pause is around 1 second long. If a pause is longer
than that, we consider that pause to be a break.

We use a couple of different methods for identifying pauses or breaks:

- We look for audio segments that are quieter than -50 db. We are sometimes more restrictive.
- We start looking 50 or more milliseconds after a speech segment. This accounts for unvoiced
  speech sounds at the end of a word, like "s".
- We look for audio segments that are untranscribable.

#### Segmentation

We segment audio based on speech segments. A speech segment starts and ends with a pause. Also,
a speech segment starts and ends on a word delimiter.

Keep in mind, that a speaker may sometimes blend words together, so we avoid segmenting at a word-
level.

The smallest segment we consider is 0.1 to 0.2 seconds. The smallest segment Google STT considers is
0.1 seconds.

#### Stops

We cut the audio such that the voice-over ends, generally, within 200 milliseconds of the last
voiced sound. The model relies on this consistency to learn when to stop generation.

#### Speed

To help monitor for errors, we like to define the slowest reading speed. In English, aside from
edge cases, this usually turns out to be around 0.2 seconds per character.

The fastest reading speed that we accept is 0.04 seconds per character.

#### Sessions

We recognize that each voice-over session has a distinct sound. Even within the same voice-over
session, a voice actor may sound different at the beginning and ending of the session.

To help increase consistency, we limit sessions to 15-minutes in length. To help the model, model
the unique aspects of each session, we give the model session context.

#### Frequency

We work in a limited range of frequencies, 20 hz and 20,000 hz based on the range of human hearing,
[learn more here](https://en.wikipedia.org/wiki/Hearing_range).

#### Yes / No Questions

To help us better model question inflection, we decided that yes / no questions should have
an upwards inflection. We have created several training and test datasets to enforce that.

#### Spectrograms

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

#### Loudness

Loudness is also difficult to define due to the peculiarities in human-hearing. Depending
on the application, we use different algorithms, including:

- RMS: This is the easiest and simplest algorithm to implement.
- ITU-R BS.1770 (LUFS): This algorithm is the industry standard.
- Custom: We calculate loudness from our spectrograms.
