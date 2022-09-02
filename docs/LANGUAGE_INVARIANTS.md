
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

We verbalize or remove non-standard words from our training data, this includes slashes,
numbers, ampersands, etc.

:eyes: Story Time! Slashes, or "us lash" have history. Awhile back, we didn't realize that slashes
can both be voiced or silent depending on the context. Short story, our model learned to stay silent
even when our users wrote out the word "slash". So, after a lot of back and forth, we figured out
that if users typed in "us lash", the model would hesitantly say "slash".
