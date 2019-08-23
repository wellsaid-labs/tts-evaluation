/**
 * - We cannot use MediaRecorder / MediaStream to implement a download button because
 * they only function to record audio while it's playing for the client:
 * https://zhirzh.github.io/2017/09/02/mediarecorder/
 */
document.addEventListener('DOMContentLoaded', function (_) {
  let clipNumber = 1;
  const endpoint = '/api/speech_synthesis/v1';

  const textarea = document.querySelector('textarea');
  const speakerSelect = document.querySelector('select');
  const apiKeyInput = document.querySelector('input');
  const generateButton = document.querySelector('#generate-button');
  const downloadAllButton = document.querySelector('#download-all');
  const errorParagraphElement = document.querySelector('#error p');
  const errorElement = document.querySelector('#error');
  const clipsElement = document.querySelector('#clips');
  const splitBySetencesInput = document.querySelector('#split-by-setences');

  const zip = (a, b) => a.map((x, i) => [x, b[i]]); // `zip` function similar to python.

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

  async function generate() {
    /**
     * Generate a clip and append it to the UI.
     */
    const textSubmitted = textarea.value.trim();
    const corpus = splitBySetencesInput.checked ?
      splitIntoSentences(textSubmitted) : [textSubmitted];
    const speakerElement = speakerSelect.options[speakerSelect.selectedIndex];
    const speakerId = parseInt(speakerElement.value);
    const apiKey = apiKeyInput.value.trim();
    const payloads = [];
    const sections = [];

    for (const text of corpus) {
      const payload = {
        speaker_id: speakerId,
        text: text.trim(),
        api_key: apiKey
      };
      payloads.push(payload);

      // Check the stream input is valid
      validateParameters(payload);

      // Create a UI element to recieve stream
      const sectionElement = document.createElement('section');
      sectionElement.innerHTML = `<main>
                                    <div>
                                      <p><b>${clipNumber}. ${speakerElement.text}</b></p>
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
      sections.push(sectionElement);
      clipsElement.prepend(sectionElement);
      clipNumber += 1;
    }

    for (const [payload, sectionElement] of zip(payloads, sections)) {
      new Promise(async () => {
        const response = await fetch(`${endpoint}/text_to_speech/input_validated`, {
          method: 'POST',
          body: JSON.stringify(payload),
          headers: {
            'Content-Type': 'application/json'
          }
        });
        if (!response.ok) {
          sectionElement.querySelector('.progress p').textContent = (await response.json()).message;
          return;
        }

        // Start stream
        let startTime = new Date().getTime();
        let queuingTime;
        const request = new XMLHttpRequest();
        let hasProgress = false;

        // Make request for stream
        request.open('POST', `${endpoint}/text_to_speech/stream`);
        request.setRequestHeader('Content-Type', 'application/json');
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
            sectionElement.querySelector('.progress p').textContent = message;
          }
        });
        request.addEventListener('error', (error) => {
          console.error(error);
          sectionElement.querySelector('.progress p').textContent = `Network error.`;
        });
        request.addEventListener('timeout', () => {
          sectionElement.querySelector('.progress p').textContent = `Request timed out.`;
        });
        request.addEventListener('abort', () => {
          sectionElement.querySelector('.progress p').textContent = `Request aborted.`;
        });
        request.addEventListener('load', () => {
          if (request.status === 200) {
            sectionElement.querySelector('.progress').style.display = 'none';
            const audioElement = sectionElement.querySelector('audio');
            const generatingTime = ((new Date().getTime()) - startTime) / 1000;

            // Add statistics to the footer on `audioElement` load.
            audioElement.addEventListener('loadedmetadata', () => {
              // `numSamples` assumes the 16-bit audio and a 44-bit WAV header.
              const numSamples = (request.getResponseHeader("Content-Length") - 44) / 2;
              const generatingWaveTime = generatingTime - queuingTime;
              const samplesPerSecond = Math.round(numSamples / generatingWaveTime);
              sectionElement.querySelector('footer p').innerHTML = ([
                `Audio Duration: ${Math.round(audioElement.duration * 100) / 100}s`,
                `Generation Timer: ${Math.round(generatingTime * 100) / 100}s`,
                `Queuing / Spectrogram Timer: ${Math.round(queuingTime * 100) / 100}s`,
                `Waveform Timer: ${Math.round(generatingWaveTime * 100) / 100}s`,
                `${Math.round(samplesPerSecond)} Samples Per Second`,
              ].join('&nbsp;&nbsp;|&nbsp;&nbsp;'));
            });

            audioElement.src = window.URL.createObjectURL(request.response);
            audioElement.load();
          } else {
            const message = `Status code ${request.status}.`;
            sectionElement.querySelector('.progress p').textContent = message;
          }
        });
        request.send(JSON.stringify(payload));
      });
    }
  }

  downloadAllButton.onclick = () => {
    /** Donwload all the audio clips on the page. */
    document.querySelectorAll('main').forEach((mainElement) => {
      const filename = mainElement.querySelector('main>div:first-child')
        .textContent.trim().replace(/\s+/g, '_').toLowerCase() + '.wav';
      const audioElement = mainElement.querySelector('audio');
      if (audioElement && audioElement.src) {
        const link = document.createElement("a");
        link.href = audioElement.src;
        link.setAttribute('download', filename);
        link.click();
      }
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
    if (isNaN(speaker_id)) {
      throw 'Speaker was not selected.';
    } else if (api_key == '') {
      throw 'API Key was not provided.';
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
