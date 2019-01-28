document.addEventListener('DOMContentLoaded', function (_) {
  const textarea = document.querySelector('textarea');
  const speakerSelect = document.querySelector('select');
  const apiKeyInput = document.querySelector('input');
  const generateButton = document.querySelector('button');
  const audioElement = document.querySelector('audio');
  const errorParagraphElement = document.querySelector('#error p');
  const errorElement = document.querySelector('#error');
  const api = '/api/speech_synthesis/v1/';

  const displayError = (message) => {
    /** Display the error element. */
    errorElement.style.display = 'block';
    errorParagraphElement.textContent = message;
  };

  const clearProgress = () => {
    /** Clear the progress from the generate button */
    generateButton.textContent = 'Generate';
  };

  const clearError = () => {
    /** Clear the error element. */
    errorElement.style.display = 'none';
  };

  function isValidParameters({
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
      displayError('Speaker was not selected.');
    } else if (api_key == '') {
      displayError('API Key was not provided.');
    } else if (text == '') {
      displayError('Text was not provided.');
    } else {
      return true;
    }
    return false;
  }

  // MediaRecorder won't work for downloading due to:
  // https://zhirzh.github.io/2017/09/02/mediarecorder/

  generateButton.onclick = async function () {
    clearError();

    // Construct the request parameters
    const parameterDictionary = {
      speaker_id: parseInt(speakerSelect.options[speakerSelect.selectedIndex].value),
      text: encodeURIComponent(textarea.value.trim()),
      api_key: apiKeyInput.value.trim(),
    }

    if (!isValidParameters(parameterDictionary)) {
      return;
    }

    const parameterString = Object.keys(parameterDictionary)
      .map(key => key + '=' + parameterDictionary[key])
      .join('&');

    // Check the stream input is valid
    const getInputValidatedURL = `${api}text_to_speech/input_validated?${parameterString}`;
    const response = await fetch(getInputValidatedURL);
    if (!response.ok) {
      const json = await response.json();
      displayError(json.message);
      return
    }

    // Start stream
    const streamURL = `${api}text_to_speech/stream?${parameterString}`;
    const request = new XMLHttpRequest();
    request.open('GET', streamURL, true);
    request.addEventListener('progress', function (event) {
      if (event.lengthComputable) {
        generateButton.textContent = `${Math.round((event.loaded / event.total) * 100)}%`;
      }
    }, false);
    request.responseType = 'blob';
    request.onreadystatechange = function () {
      if (request.readyState === 4 && request.status === 200) {
        audioElement.src = window.URL.createObjectURL(request.response);
        audioElement.load();
        clearProgress();
      }
    };
    request.send();
  };
});
