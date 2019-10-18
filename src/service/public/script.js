/**
 * - We cannot use MediaRecorder / MediaStream to implement a download button because
 * they only function to record audio while it's playing for the client:
 * https://zhirzh.github.io/2017/09/02/mediarecorder/
 */
document.addEventListener('DOMContentLoaded', function (_) {
  let clipNumber = 1;
  // NOTE: This endpoint is backwards compatible.
  const endpoint = '/api/speech_synthesis/v1/text_to_speech';

  const textarea = document.querySelector('textarea');
  const speakerSelect = document.querySelector('#speaker');
  const speakerOptionElements = document.querySelectorAll('#speaker option');
  const apiKeyInput = document.querySelector('input');
  const generateButton = document.querySelector('#generate-button');
  const downloadAllButton = document.querySelector('#download-all');
  const errorParagraphElement = document.querySelector('#error p');
  const errorElement = document.querySelector('#error');
  const clipsElement = document.querySelector('#clips');
  const splitBySetencesInput = document.querySelector('#split-by-setences');
  const versionSelect = document.querySelector('#version');
  const allAudio = [];

  versionSelect.onchange = () => {
    const version = versionSelect.value.toLowerCase();
    let speakers;
    // TODO: Share a configuration file between server and browser on the versions available
    // and the speakers available.
    if (version == "v1") {
      speakers = [0, 1, 2, 3];
    } else if (version == "v2" || version == "latest") {
      speakers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    }
    speakerOptionElements.forEach(option => {
      option.disabled = !speakers.includes(parseInt(option.value));
    });
  }

  // `zip` function similar to python.
  const zip = (...args) => args[0].map((_, i) => args.map(arg => arg[i]));

  /**
   * Sleep for a number of milliseconds.
   *
   * @param {number} milliseconds Number of milliseconds to sleep.
   * @returns {Promise}
   */
  function sleep(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
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

  function requestAudio(data) {
    // Start stream
    let startTime = new Date().getTime();
    let queuingTime;
    const request = new XMLHttpRequest();
    let hasProgress = false;

    // Make request for stream
    request.open('POST', `https://voice.wellsaidlabs.com${endpoint}/stream`);
    request.setRequestHeader('Content-Type', 'application/json');
    request.setRequestHeader('Accept-Version', data.version);
    request.responseType = 'blob';
    request.addEventListener('progress', (event) => {
      if (event.lengthComputable) {
        if (!hasProgress && event.loaded > 0) {
          hasProgress = true;
          // The server sends first WAV headers, before generating the spectrogram.
          queuingTime = ((new Date().getTime()) - startTime) / 1000;
        }
        let percentage = event.loaded / event.total;
        const now = new Date().getTime();
        let estimatedTimeLeft = ((now - startTime) / percentage) * (1 - percentage);
        estimatedTimeLeft = secondsToString(estimatedTimeLeft / 1000);
        percentage = Math.round(percentage * 100); // Decimals to percent
        const message = `${Math.round(percentage)}% Generated - ${estimatedTimeLeft} Left`
        data.sectionElement.querySelector('.progress p').textContent = message;
      }
    });
    request.addEventListener('error', (error) => {
      console.error(error);
      data.sectionElement.querySelector('.progress p').textContent = `Network error.`;
    });
    request.addEventListener('timeout', () => {
      data.sectionElement.querySelector('.progress p').textContent = `Request timed out.`;
    });
    request.addEventListener('abort', () => {
      data.sectionElement.querySelector('.progress p').textContent = `Request aborted.`;
    });
    request.addEventListener('load', () => {
      if (request.status === 200) {
        data.sectionElement.querySelector('.progress').style.display = 'none';
        const audioElement = data.sectionElement.querySelector('audio');
        const generatingTime = ((new Date().getTime()) - startTime) / 1000;

        // Add statistics to the footer on `audioElement` load.
        audioElement.addEventListener('loadedmetadata', () => {
          // `numSamples` assumes the 16-bit audio and a 44-bit WAV header.
          const numSamples = (request.getResponseHeader("Content-Length") - 44) / 2;
          const generatingWaveTime = generatingTime - queuingTime;
          const samplesPerSecond = Math.round(numSamples / generatingWaveTime);
          data.sectionElement.querySelector('footer p').innerHTML = ([
            `Audio Duration: ${Math.round(audioElement.duration * 100) / 100}s`,
            `Generation Timer: ${Math.round(generatingTime * 100) / 100}s`,
            `Queuing / Spectrogram Timer: ${Math.round(queuingTime * 100) / 100}s`,
            `Waveform Timer: ${Math.round(generatingWaveTime * 100) / 100}s`,
            `${Math.round(samplesPerSecond)} Samples Per Second`,
          ].join('&nbsp;&nbsp;|&nbsp;&nbsp;'));
        });

        allAudio.push({
          'response': request.response,
          'speakerId': data.payload.speaker_id,
          'text': data.payload.text,
          'clipNumber': data.clipNumber,
        });
        audioElement.src = window.URL.createObjectURL(request.response);
        audioElement.load();
      } else {

        const message = `Status code ${request.status}.`;
        data.sectionElement.querySelector('.progress p').textContent = message;
      }
    });
    request.send(JSON.stringify(data.payload));
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
    const requests = [];

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
                                      <div class="progress">
                                        <p>Queuing / Generating Spectrogram</p>
                                      </div>
                                      <audio controls></audio>
                                    </div>
                                  </main>
                                  <footer>
                                    <p></p>
                                  </footer>`;
      clipsElement.prepend(sectionElement);
      clipNumber += 1;

      try {
        const response = await fetch(`https://voice2.wellsaidlabs.com${endpoint}/input_validated`, {
          method: 'POST',
          body: JSON.stringify(payload),
          headers: {
            'Content-Type': 'application/json',
            'Accept-Version': version,
          }
        })
        if (!response.ok) {
          sectionElement.querySelector('.progress p').textContent = (await response.json()).message;
        } else {
          requests.push({
            version,
            clipNumber: clipNumber - 1,
            payload,
            sectionElement,
          });
        }
      } catch (error) {
        // TODO: Retry input validation a couple times if so.
        console.error(error);
        sectionElement.querySelector('.progress p').textContent = 'Network error';
      }
    }

    for (const request of requests) {
      // NOTE: Chrome will only keep six connections open at a time:
      // https://stackoverflow.com/questions/29206067/understanding-chrome-network-log-stalled-state
      requestAudio(request);
      // NOTE: Ensure that the requests are captured by the server in the right order. In the
      // future, this can be done more efficiently with a `Date` header. Learn more:
      // https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Date
      await sleep(100);
    }
  }

  downloadAllButton.onclick = () => {
    /** Donwload all the audio clips on the page. */
    if (allAudio.length == 0) {
      return;
    }

    const zip = new JSZip();
    const clips = zip.folder("clips");
    allAudio.forEach(function ({
      response,
      speakerId,
      text,
      clipNumber,
      maxTextLength = 50, // NOTE: There is a maximum length in various OSs for filenames.
    }) {
      const speakerName = speakerSelect.options[speakerId].text;
      let partialText = text.trim();
      if (partialText.length > maxTextLength) {
        partialText = partialText.slice(0, maxTextLength) + '...';
      }
      clips.file(clipNumber + ' - ' + speakerName.trim() + ' - ' + partialText + '.wav', response);
    });
    zip.generateAsync({
      type: "blob"
    }).then(content => {
      saveAs(content, 'clips.zip');
    });
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
});
