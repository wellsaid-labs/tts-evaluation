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
  const generateButton = document.querySelector('button');
  const errorParagraphElement = document.querySelector('#error p');
  const errorElement = document.querySelector('#error');
  const clipsElement = document.querySelector('#clips');
  const splitBySetencesInput = document.querySelector('#split-by-setences');

  // `zip` function similar to python.
  const zip = (a, b) => a.map((x, i) => [x, b[i]]);

  async function generate() {
    /**
     * Generate a clip and append it to the UI.
     */
    const textSubmitted = textarea.value.trim();
    const corpus = splitBySetencesInput.checked ? sentenceSplit(textSubmitted) : [textSubmitted];
    const speakerElement = speakerSelect.options[speakerSelect.selectedIndex];
    const speakerId = parseInt(speakerElement.value);
    const apiKey = apiKeyInput.value.trim();
    const payloads = [];
    const sections = [];

    for (const text of corpus) {
      const payload = {
        speaker_id: speakerId,
        text: text,
        api_key: apiKey
      };
      payloads.push(payload);

      // Check the stream input is valid
      validateParameters(payload);
      const response = await fetch(`${endpoint}/text_to_speech/input_validated`, {
        method: 'POST',
        body: JSON.stringify(payload),
        headers: {
          'Content-Type': 'application/json'
        }
      });
      if (!response.ok) {
        throw (await response.json()).message;
      }

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

    const promises = [];
    for (const [payload, sectionElement] of zip(payloads, sections)) {
      promises.push(new Promise(function (resolve, _) {
        // Start stream
        let startTime = new Date().getTime();
        let queuingTime;
        const request = new XMLHttpRequest();
        let hasProgress = false;

        // Make request for stream
        request.open('POST', `${endpoint}/text_to_speech/stream`);
        request.setRequestHeader('Content-Type', 'application/json');
        request.responseType = 'blob';
        request.addEventListener('loadstart', resolve);
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
      }));
    }
    return Promise.all(promises);
  }

  let isGenerateButtonDisabled = false;
  generateButton.onclick = async function () {
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

  function sentenceSplit(text) {
    /**
     * Inspired by:
     * https://stackoverflow.com/questions/11761563/javascript-regexp-for-splitting-text-into-sentences-and-keeping-the-delimiter
     *
     * @param {string} text
     * @returns {Array.<string>}
     */
    return text.match(/([^\.!\?]+[\.!\?]+)|([^\.!\?]+$)/g);
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
