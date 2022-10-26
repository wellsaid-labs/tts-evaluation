# <speak>This output speech uses SSML.</speak>
# If we put the SML parsing in _data, should it parse anything?
# Does it handle respellings also?
# Should the interface be SSML?
# Why are respellings in `inputs`? Why not in `data`?
# Maybe?
# preprocess_span is called in the model
# Should the input to the model be XML? Probably? Where else would it be?
# The model recieves text directly
# So, maybe, that's it? We have the model parse it's own XML? And then we also generate it?

# How would this work with timepoints, well, you mark it up, and the model outputs the attention
# as needed?
# What is the altnerative? The model takes another input and we need to carry around a
# preprocessing and postprocessing elemtn?

# NOTE: At the moment, we do pass a `doc` element throughout the code base. Have we considered
# passing thorugh a different element? Like a `span`? Yeah, lol, we've considered that.
# Should the span have these annotations added to it?
# The `span` is usually not involed in training? Having it generate random masks, would be,
# unusual?

# The current question that I have is... how do we want to handle pronunciations? So, say,
# we've been passed in some XML?
# Well, ideally, spaCy would be fast enough to run

# Hmm... instead of passing around XML, cause, like, why? Why would we want to use that format
# as the interface? It's clunky.
# Should the model accept a different interface? Maybe it accepts a doc object, with
# the related loudness, loudness mask, speed, speed mask, etc...

# AND then that object has a TO_XML, FROM_XML?
# I mean... okay, we can create it directly, or we can create it from_xml. The model can directly
# integrate it as part of it's input?
# What I do not like is that model inputs being defined in `inputs.py`? Why do I not like that?
# I think what I do not like is that respellings are choosen in the model inputs? I think
# I do not like that public errors are deep inside the ML code base? Why is this considered deep?
# I think, ideally, there would be a cleaner interface for validating the respelling a bit earlier
# in the process?

# TODO: Create an object called
# `Inputs`
# - Loudess [(start, stop, loudness)]
# - Rate [(start, stop, rate)]
# - Respelling [(start, stop, respelling)]
# - Session
# - Doc
# - Context (Optional)
# `PreprocessedInputs` (basically the current `Inputs` element)

# TODO: We probably do not need `prefix`, `suffix`, `delim` because we are not replacing
# the respelling and tokenizing, again, right?
# TODO: We do not need `norm_respellings` because we again are not dealing with a string.
# TODO:

# NOTE: The current implementation of contextual training does not have a good parallel to
# inference. Should it?
# TODO: If we have in context text, why not contextual audio too? Heck, why do we not iterate
# from left-to-right accross an entire dataset, similar to, how the voice actor read through it?
# A consideration we would need to make is if the voice-actor made a mistake.
# TODO: If we decide that contextual read throughs like the above are not the right direction,
# should we remove context all together? Should we try to find text that can be read without
# any additional context? For example, we could consider training on full sentences, only?


# TODO: How are we going to think about `session`s when going `from_xml` and `to_xml`?
# We need some sort of vocabularly... :(
# If we have the `to_xml` and `from_xml`, here, then it can easily be adjusted for
#
# The two spots are want to use this are:
# - We want to grab XML and parse it from the worker? Unfortunately, that cannot
# happen without a session map, we can provide that.
# - We also want to generate XML that can be used otherwise? We could provide
# a session map then also?
# - Now, should, the model be parsing the XML? Like why is this accepting a map
# to help with the parsing? Should things be normalized before it gets here? Ideally,
# yes. Unfortunately, we cannot make Sessions easily normalized? Unless we really
# go out of our way to ask folks to include the full XML object? There might be benefit
# to that? People would be able to craft their own speakers? The problem is that...
# The session names are sesnitive. Like, really sensitive, which is a problem.
# They are in the audio names, so, I'd think about... not asking users to submit those.
# I think this is an interesting idea, and a good TODO to include
# TODO: Implement


# TODO: Double check invariants...
# Respellings need to line up with tokens? Are these the right datatypes? I think so,
# these, are the easiest, to create? Except maybe, respellings? But even those,
# are pretty easy? Because there is an issue...
# We could do a map but then we'd need to preprocess with spacy in `from_xml`
# Well, we'd need to verbalize too. That complicates things. How do we verbalize?
# Whilst ignoring the XML?
# Shoot... That's pretty hard.
# actually, it's not that bad... the verbalization works with matches, right?
# They shoudl be SORTED
# They should not have overlaps
# No annotations in context, yet?
# The annotations do not care about the context, so their slices are not offset by
# it? Maybe this is a __post_init__ process to adust the annotation slices?? I think so.


from lxml import etree as ET

xml = """
<speak value="1">
  <loudness value="-20">Over the river and
  <tempo value="0.04">through the <respell value="wuuds">woods</respell>
  </tempo>.</loudness>
</speak>
"""


xml_schema_doc = ET.parse("run/_models/spectrogram_model/schema.xsd", None)
xml_schema = ET.XMLSchema(xml_schema_doc)

root = ET.fromstring(xml, None)
try:
    xml_schema.assertValid(root)
except ET.DocumentInvalid as xml_errors:
    print("List of errors:\r\n", xml_errors.error_log)
    exit()

parser = ET.XMLPullParser(events=("start", "end"))
parser.feed(xml)

loudness = []
loudness_mask = []
speed = []
speed_mask = []
pronunciations = []

for event, elem in parser.read_events():
    if event == "start":
        print("tag:", elem.tag)  # use only tag name and attributes here
        if elem.tag == "prosody":
            print(elem.get("loundess"))
        if elem.text:
            print(repr(elem.text))
    elif event == "end":
        print(elem.tag)
        # elem children elements, elem.text, elem.tail are available
        if elem.tail:
            print(repr(elem.tail))
