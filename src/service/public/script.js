/**
 * - We cannot use MediaRecorder / MediaStream to implement a download button because
 * they only function to record audio while it's playing for the client:
 * https://zhirzh.github.io/2017/09/02/mediarecorder/
 */
document.addEventListener('DOMContentLoaded', async function (_) {
  let clipNumber = 1;

  // NOTE: This endpoint is backwards compatible.
  const endpoint = '/api/speech_synthesis/v1/text_to_speech';
  const is_production = location.hostname.includes('wellsaidlabs');

  const textarea = document.querySelector('textarea');
  const speakerSelect = document.querySelector('#speaker');
  const speakerOptionElements = document.querySelectorAll('#speaker option');
  const apiKeyInput = document.querySelector('input');
  const generateButton = document.querySelector('#generate-button');
  const errorParagraphElement = document.querySelector('#error p');
  const errorElement = document.querySelector('#error');
  const clipsElement = document.querySelector('#clips');
  const splitBySetencesInput = document.querySelector('#split-by-setences');
  const versionSelect = document.querySelector('#version');
  const allRequests = [];

  function filterSpeakers() {
    const version = versionSelect.value.toLowerCase();
    let speakers;
    // TODO: Share a configuration file between server and browser on the versions available
    // and the speakers available.
    if (version == "v1") {
      speakers = [...Array(4).keys()];
    } else if (version == "v2") {
      speakers = [...Array(12).keys()];
    } else if (version == "v3" || version == "v4" || version == "v5" ||
      version == "v6" || version == "latest") {
      speakers = [...Array(20).keys()];
    } else if (version == "lincoln.v1") {
      speakers = [11541];
    }
    speakerOptionElements.forEach(option => {
      option.disabled = !speakers.includes(parseInt(option.value));
    });
  }

  filterSpeakers(); // Initial filter
  versionSelect.onchange = filterSpeakers;

  /**
   * Sleep for a number of milliseconds.
   *
   * @param {number} milliseconds Number of milliseconds to sleep.
   * @returns {Promise}
   */
  function sleep(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
  }

  function roundHundredth(number) {
    return Math.round(number * 100) / 100;
  }

  function splitIntoSentences(text) {
    /**
     * Inspired by:
     * https://stackoverflow.com/questions/11761563/javascript-regexp-for-splitting-text-into-sentences-and-keeping-the-delimiter
     * https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing
     * https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
     */
    if (text.includes("\n")) {
      return text.split("\n").map(t => splitIntoSentences(t)).flat();
    }

    text = text.trim();
    text = text.replace(/([0-9]+)[.]([0-9]+)/, '$1<period>$2');
    text = text.replace(/(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt|Inc|Ltd|Jr|Sr|Co)[.]/g, '$1<period>');
    text = text.replace(/Ph\.D\./g, 'Ph<period>D<period>');
    text = text.replace(/e\.g\./g, 'e<period>g<period>');
    text = text.replace(/i\.e\./g, 'i<period>e<period>');
    text = text.replace(/vs\./g, 'vs<period>');
    text = text.replace(/([A-Za-z])[.]([A-Za-z])[.]([A-Za-z])[.]/g, '$1<period>$2<period>$3<period>');
    text = text.replace(/([A-Za-z])[.]([A-Za-z])[.]/g, '$1<period>$2<period>');
    text = text.replace(/[.](com|net|org|io|gov|me|edu)/g, '<period>$1');
    text = text.replace(/([.!?][.!?"”'’)]{0,})/g, '$1<stop>');
    text = text.replace(/<period>/g, '.');
    return text.split('<stop>').map(t => t.trim()).filter(s => s.length > 0);
  }

  console.assert(splitIntoSentences('A woman screams .').length == 1);
  console.assert(splitIntoSentences('A woman screams . Shrill and desolate, the brief sound rips ' +
    'through the solemn hush in the corridors of the federal courthouse.').length == 2);
  console.assert(splitIntoSentences('“No pulse,” the doctor shouts.  ' +
    '“Help, please!” ').length == 2);
  console.assert(splitIntoSentences('Marta would say ... A woman screams .').length == 2);
  console.assert(splitIntoSentences('You will learn that Kiril Pafko is both a medical ' +
    'doctor, an M.D., and a Ph.D. in biochemistry. ').length == 1);
  console.assert(splitIntoSentences('Mr. John Johnson Jr. was born in the U.S.A but earned his ' +
    'Ph.D. in Israel before joining Nike Inc. as an engineer. He also worked at craigslist.org ' +
    'as a business analyst.').length == 2);
  console.assert(splitIntoSentences('Good morning Dr. Adams. The patient is ' +
    'waiting for you in room number 3.').length == 2);
  console.assert(splitIntoSentences('This is waaaaayyyy too much for you!!!!!! ' +
    'This is waaaaayyyy too much for you!!!!!!').length == 2);
  console.assert(splitIntoSentences('(How does it deal with this parenthesis?)  ' +
    '"It should be a new sentence." "(And the same with this one.)" ' +
    '(\'And this one!\')"(\'(And (this)) \'?)"').length == 5);

  /**
   * Request `data.sectionElement` to be populated with an audio stream.
   *
   * @param {Object} data
   * @param {String} data.version Audio request version header.
   * @param {Object} data.payload Audio request payload.
   * @param {HTMLElement} data.sectionElement The HTMLElement to be populated.
   */
  function requestAudio(data) {
    return new Promise((resolve) => {
      const audioElement = data.sectionElement.querySelector('audio');
      // TODO: A GET request traditionally does not support more than 2,000 characters. We might
      // need to use a POST request.
      // TODO: Enable downloading the audio clips.
      // TODO: Print the errors to the user instead of console.
      // NOTE: This is largely inspired by this example:
      // https://developers.google.com/web/fundamentals/media/mse/basics
      const startTime = new Date().getTime();
      const audioSource = `${is_production ? 'https://voice.wellsaidlabs.com' : ''}${
        endpoint}/stream?${new URLSearchParams(data.payload).toString()}`;
      const mediaSource = new MediaSource();
      audioElement.src = URL.createObjectURL(mediaSource);
      mediaSource.addEventListener('sourceopen', async (event) => {
        URL.revokeObjectURL(audioElement.src);
        const response = await fetch(audioSource, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
            'Accept-Version': data.version,
          }
        }).catch((error) => {
          console.error('Error:', error);
        });
        const ttfbTime = new Date().getTime(); // Time To First Byte (TTFB)
        const reader = response.body.getReader();
        const mediaSource = event.target;
        const sourceBuffer = mediaSource.addSourceBuffer(response.headers.get('Content-Type'));
        const onEndOfStream = () => {
          if (!sourceBuffer.updating && mediaSource.readyState === 'open') {
            mediaSource.endOfStream();
          }
        }
        const getBuffered = () => { // Get the maximum buffer time
          let maxBuffered = 0;
          for (let i = 0; i < sourceBuffer.buffered.length; i++) {
            maxBuffered = Math.max(sourceBuffer.buffered.end(i), maxBuffered);
          }
          return maxBuffered;
        }
        while (true) {
          let {
            done,
            value,
          } = await reader.read();
          if (done) {
            break;
          }
          sourceBuffer.appendBuffer(value.buffer);
          sourceBuffer.addEventListener('update', () => {
            if (!sourceBuffer.updating && mediaSource.readyState === 'open') {
              mediaSource.duration = getBuffered();
            }
          });
          const generationTime = (new Date().getTime() - ttfbTime) / 1000;
          data.sectionElement.querySelector('footer p').innerHTML = ([
            `Queuing Time: ${roundHundredth((ttfbTime - startTime) / 1000)}s`,
            `Real Time: ${roundHundredth(generationTime / getBuffered())}x`,
          ].join('&nbsp;&nbsp;|&nbsp;&nbsp;'));
        }
        onEndOfStream();
        sourceBuffer.addEventListener('updateend', onEndOfStream);
      });
      resolve();
    });
  };

  async function generate() {
    /**
     * Generate a clip and append it to the UI.
     */
    const textSubmitted = textarea.value.trim();
    const corpus = splitBySetencesInput.checked ?
      splitIntoSentences(textSubmitted) : [textSubmitted];
    const speakerElement = speakerSelect.options[speakerSelect.selectedIndex];
    const versionElement = versionSelect.options[versionSelect.selectedIndex];
    const speakerId = parseInt(speakerElement.value);
    const version = versionElement.value;
    const apiKey = apiKeyInput.value.trim();

    for (const text of corpus) {
      const payload = {
        speaker_id: speakerId,
        text: text.trim(),
        api_key: apiKey
      };

      // Check the stream input is valid
      validateParameters(payload);

      // Create a UI element to recieve stream
      const sectionElement = document.createElement('section');
      sectionElement.innerHTML = `<main>
                                    <div>
                                      <p><b>${clipNumber}. ${speakerElement.text} (${
                                        version})</b></p>
                                      <p>${text}</p>
                                    </div>
                                    <div>
                                      <audio controls></audio>
                                    </div>
                                  </main>
                                  <footer>
                                    <p></p>
                                  </footer>`;
      clipsElement.prepend(sectionElement);
      clipNumber += 1;

      try {
        const response = await fetch(
          `${is_production ? 'https://voice2.wellsaidlabs.com' : ''}${endpoint}/input_validated`, {
            method: 'POST',
            body: JSON.stringify(payload),
            headers: {
              'Content-Type': 'application/json',
              'Accept-Version': version,
            }
          })
        if (!response.ok) {
          console.error((await response.json()).message);
        } else {
          allRequests.push({
            version,
            payload,
            sectionElement,
          });
        }
      } catch (error) {
        // TODO: Retry input validation a couple times if so.
        console.error(error);
      }
    }
  }

  let isGenerateButtonDisabled = false;
  generateButton.onclick = async () => {
    if (!isGenerateButtonDisabled) {
      clearError();
      disableGenerateButton();
      try {
        await generate();
      } catch (error) {
        displayError(error);
      }
      enableGenerateButton();
    }
  };

  function validateParameters({
    speaker_id,
    text,
    api_key
  }) {
    /**
     * Validate the input for audio generation.
     *
     * @param {string} speaker_id
     * @param {string} text
     * @param {string} api_key
     */
    if (api_key == '') {
      throw 'API Key was not provided.';
    } else if (isNaN(speaker_id)) {
      throw 'Speaker was not selected.';
    } else if (text == '') {
      throw 'Text was not provided.';
    } else {
      return;
    }
  }


  function displayError(message) {
    /** Display the error element. */
    errorElement.style.display = 'block';
    errorParagraphElement.textContent = message;
  };

  function clearError() {
    /** Clear the error element. */
    errorElement.style.display = 'none';
  };

  function disableGenerateButton() {
    /** Clear the error element. */
    isGenerateButtonDisabled = true;
    generateButton.innerHTML = '<div class="loader"></div>';
  };

  function enableGenerateButton() {
    /** Clear the error element. */
    isGenerateButtonDisabled = false;
    generateButton.innerHTML = 'Generate';
  };

  function secondsToString(seconds) {
    seconds = Math.round(seconds);
    const hours = Math.floor(seconds / (60 * 60));

    const divisor_for_minutes = seconds % (60 * 60);
    const minutes = Math.floor(divisor_for_minutes / 60);

    const divisor_for_seconds = divisor_for_minutes % 60;
    seconds = Math.ceil(divisor_for_seconds);

    if (hours > 0) {
      return `${hours}hr`;
    } else if (minutes > 0) {
      return `${minutes}m`;
    } else {
      return `${seconds}s`;
    }
  }

  (async () => {
    while (true) {
      if (allRequests.length > 0) {
        // NOTE: To ensure that requests are completed by the server, wait for a request to start
        // streaming before requesting again. Otherwise, Chrome might timeout because it's been
        // waiting on the request to complete for too long.
        await requestAudio(allRequests.shift());
      }
      await sleep(100);
    }
  })();
});
