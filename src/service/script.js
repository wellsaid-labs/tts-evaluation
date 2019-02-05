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
                                    <p>0% Generated</p>
                                  </div>
                                  <audio controls></audio>
                                </div>`;
    clipsElement.prepend(sectionElement);
    clipNumber += 1;

    return new Promise(function (resolve, reject) {
      // TODO: Use the reject parameter and resolve if loadstart does not work.
      // Start stream
      const streamURL = `${endpoint}/text_to_speech/stream?${parameterString}`;
      const request = new XMLHttpRequest();

      // Make request for stream
      request.open('GET', streamURL);
      request.responseType = 'blob';
      request.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          resolve();
          const percentage = Math.round((event.loaded / event.total) * 100);
          sectionElement.querySelector('.progress p').textContent = `${percentage}% Generated`;
        }
      });
      request.addEventListener('error', (error) => {
        console.error(error);
        sectionElement.querySelector('.progress p').textContent = `Unknown error, please retry.`;
      });
      request.addEventListener('timeout', () => {
        sectionElement.querySelector('.progress p').textContent = `Request timedout.`;
      });
      request.addEventListener('abort', () => {
        sectionElement.querySelector('.progress p').textContent = `Request aborted.`;
      });
      request.addEventListener('load', (error) => {
        if (request.status === 200) {
          sectionElement.querySelector('.progress').style.display = 'none';
          sectionElement.querySelector('audio').src = window.URL.createObjectURL(request.response);
          sectionElement.querySelector('audio').load();
        } else {
          sectionElement.querySelector('.progress p').textContent = `${request.status} error.`;
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
});
