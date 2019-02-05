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

  async function generate() {
    /**
     * Generate a clip and append it to the UI.
     */
    const speakerElement = speakerSelect.options[speakerSelect.selectedIndex];
    const textSubmitted = textarea.value.trim();
    const parameterDictionary = {
      speaker_id: parseInt(speakerElement.value),
      text: encodeURIComponent(textSubmitted),
      api_key: apiKeyInput.value.trim(),
    }
    validateParameters(parameterDictionary);

    const parameterString = Object.keys(parameterDictionary)
      .map(key => key + '=' + parameterDictionary[key])
      .join('&');

    // Check the stream input is valid
    const getInputValidatedURL = `${endpoint}/text_to_speech/input_validated?${parameterString}`;
    const response = await fetch(getInputValidatedURL);
    if (!response.ok) {
      throw (await response.json()).message;
    }

    // Create a UI element to recieve stream
    const sectionElement = document.createElement('section');
    sectionElement.innerHTML = `<div>
                                  <p><b>${clipNumber}. ${speakerElement.text}</b></p>
                                  <p>${textSubmitted}</p>
                                </div>
                                <div>
                                  <div class="progress">
                                    <p>Queuing</p>
                                  </div>
                                  <audio controls></audio>
                                </div>`;
    clipsElement.prepend(sectionElement);
    clipNumber += 1;

    return new Promise(function (resolve, _) {
      // TODO: Use the reject parameter and resolve if loadstart does not work.
      // Start stream
      let startTime = new Date().getTime();
      const streamURL = `${endpoint}/text_to_speech/stream?${parameterString}`;
      const request = new XMLHttpRequest();

      // Make request for stream
      request.open('GET', streamURL);
      request.responseType = 'blob';
      request.addEventListener('loadstart', resolve);
      request.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
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
          sectionElement.querySelector('audio').src = window.URL.createObjectURL(request.response);
          sectionElement.querySelector('audio').load();
        } else {
          const message = `Status code ${request.status}.`;
          sectionElement.querySelector('.progress p').textContent = message;
        }
      });
      request.send();
    });
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
