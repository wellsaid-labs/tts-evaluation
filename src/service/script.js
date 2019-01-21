// TODO: If error, write error to html.

const textarea = document.querySelector("textarea");
const speakerSelect = document.querySelector("select");
const securityTokenInput = document.querySelector("input");
const generateButton = document.querySelector("button");
const audioElement = document.querySelector("audio");

generateButton.onclick = async function () {
  var parameters = {
    speaker_id: parseInt(speakerSelect.options[speakerSelect.selectedIndex].value),
    text: encodeURIComponent(textarea.value.trim()),
    token: securityTokenInput.value.trim(),
  }
  parameters = Object.keys(parameters).map(key => key + '=' + parameters[key]).join('&');
  const audioSource = `/api/speech_synthesis/v1/text_to_speech?${parameters}`
  audioElement.src = audioSource;
  audioElement.preload = 'metadata'
  audioElement.type = 'audio/wav'
  audioElement.load();
  // Although it may not be loaded, this stops the `audioElement` from loading again if the user
  // presses play.
  audioElement.play();
};
