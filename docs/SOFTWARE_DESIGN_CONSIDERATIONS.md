# Software Design Considerations  :nail_care:

At a high-level, here are some of the considerations we made when designing this
software. To help keep this software consistent, please adhere to these guidelines

## Architecture

["Functional Core, Imperative Shell"](https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell)
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

Within run, we categorize our modules. The modules with underscores `_` are not runnable, they
are utility modules.

For us it's particularly important to document our assumptions, we do so in the `_config` module.
This module allows us to configure hyperparameters across the board from our data processing
to training.

### Library `lib`

The functions in `lib` are reliable and well-tested as they constitute a functional core that
the rest of the application relies on.

### Third Party `third_party`

This is a great place to throw any copy-pasted code from external repositories.

## Dataset Invariants

To model text-to-speech, we rely on our datasets to be consistent. We assume that there is a
one-to-one transformation from text to speech that the model can learn. Our hope is to create
a text-to-speech that is akin to a piano, it's intuitive to use and beautiful to listen to.

With that in mind, we have defined a couple of invariants that we rely on...

- [Language Invariants](docs/LANGUAGE_INVARIANTS.md)
- [Voice-Over Invariants](docs/VOICE_OVER_INVARIANTS.md)

*Keep in mind that many of these invariants were created for English*
