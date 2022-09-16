
# Language Invariants

While building this software, we have thought about language as a tool for creating voice-over.
In doing so, we have restricted some of the ways we interpret written text.

## Progression :chart_with_upwards_trend:

We have restricted our model to read left-to-right without skipping around the text. To support
this, we have also normalized our scripts to verbalize phrases like "$5 million" which is
read "five million dollars" instead of "dollar five million".

With that in mind, we expect that the voice-over stops after the last letter is read by the model.

## Letter Casing

We have decided, in order to reduce ambiguity, that uppercase letters should only be used to
represent initialisms. For example, we remove non-initialisms by comparing the letter casing in the
transcript with the voice-over script. Also, we consider acronyms to be non-standard words that are
verbalized into standard words (i.e. "NCAA" to "NC double A"). We have also created datasets that
have only initialisms and no acronyms to help enforce this invariant.

## Alphabet :a:

We have decided to restrict the alphabet to only ASCII characters, with a few exceptions.
The terminology that we like to use internally is "readable characters". We hope that
the alphabet closely aligns with the characters voice-actors can read. This also includes
normalizing some rarely used characters like backticks, curly braces, etc.

## Non-Standard Words

Our model does not support non-standard words. In order to handle those, we verbalize
this data during inference and we remove non-standard words during training. These
include: slashes, numbers, ampersands, etc. We ask the our script writers do not use non-standard
words in their scripts.

:eyes: Story Time! Slashes, or "us lash" have history. Awhile back, we didn't realize that slashes
can both be voiced or silent depending on the context. Short story, our model learned to stay silent
even when our users wrote out the word "slash". So, after a lot of back and forth, we figured out
that if users typed in "us lash", the model would hesitantly say "a slash". This was a helpful solution in one case, but did not solve the universal need for a pronunciation of "slash" as a single word in other contexts.

## Scripts

We generally expect a voice-over script to be at most a couple paragraphs in length, and we refer
to this as a "Passage". There needs to be multiple "Passages" in a dataset. A "Passage" should be
one consistent piece of writing on a single topic.

This is a constraint for a couple of reasons. First of all, the model is unable to model long-range
context, so any more will not be helpful during training. This is also helpful for performance.
It's much easier to pre-process many shorter passages rather than one longer passage. Lastly, we
need to split the dataset for evaluation purposes. In order to do that, there needs to be
multiple scripts, some that can be used for evaluation and some that can be used for training.

Additionally, we've had issues in the past, that long scripts are, in reality, are multiple scripts
that have been accidentally added together.

## Pronunciation

We assume that words have different pronunciations based on context and speaker. This is due
to dialect, part-of-speech, and usage (i.e. "Bass", as a noun, could refer to a fish or sound).

For this reason, we need to provide adequate context for the pronunciation to be correct. We rely
on extensive dictionaries to build respellings that support multiple dialects.
