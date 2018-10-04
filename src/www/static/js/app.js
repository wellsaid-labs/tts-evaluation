$(document).ready(function () {

  // Modal
  var $modal = $('#modal');
  var $closeButton = $('.close')
  var $listenButton = $('#listen-button');

  function displayModal(speaker, filename) {
    $modal.show();
    $closeButton.fadeIn(3000);

    disableListenButton();

    // Update speaker row
    var $avatar = $('.modal .speaker-row-modal img');
    $avatar.attr('src', 'static/images/avatar_' + speaker.toLowerCase() + '.png');
    $('.modal .speaker-details-modal h3').text(speaker);

    // Update audio file
    var $audioPlayer = $('#player');
    var $audioSource = $('#player source');
    $audioSource.attr('src', '../static/samples/' + filename);
    $audioPlayer[0].pause();
    $audioPlayer[0].load();

    // Update play button state
    $('#play-button').addClass('pause');

    // Update download link
    $('#download-link').attr('href', '../samples/' + filename + '?attachment=True');
  }

  function closeModal() {
    var $audioPlayer = $('#player');
    $audioPlayer[0].pause();
    $audioPlayer[0].currentTime = 0;
    $modal.hide();
    $closeButton.hide();
    enableListenButton();
  }

  $closeButton.on('click', function (event) {
    closeModal();
  });

  function disableListenButton() {
    $listenButton.prop('disabled', true);
    $listenButton.text('Listen');
    $listenButton
      .removeClass('btn-enabled')
      .addClass('btn-disabled');
  }

  function enableListenButton() {
    $listenButton.prop('disabled', false);
    $listenButton.text('Listen');
    $listenButton
      .removeClass('btn-disabled')
      .addClass('btn-enabled');
  }

  function calculateTotalValue(length) {
    var minutes = Math.floor(length / 60),
      seconds_int = length - minutes * 60,
      seconds = Math.round(seconds_int);

    seconds = (seconds < 10 ? '0' + seconds : seconds);
    time = (minutes < 1 ? ('00:' + seconds) : (minutes + ':' + seconds));

    return time;
  }

  function calculateCurrentValue(currentTime) {
    var current_hour = parseInt(currentTime / 3600) % 24,
      current_minute = parseInt(currentTime / 60) % 60,
      current_seconds_long = currentTime % 60,
      current_seconds = current_seconds_long.toFixed(),
      current_time =
        (current_minute < 1 ? '0' + current_minute : current_minute) +
        ':' +
        (current_seconds < 10 ? '0' + current_seconds : current_seconds);

    return current_time;
  }

  function initProgressBar() {
    var player = document.getElementById('player');
    var length = player.duration;
    var current_time = player.currentTime;

    if (!isNaN(length)) {
      var totalLength = calculateTotalValue(length);
      $('.end-time').text(totalLength);

      var currentTime = calculateCurrentValue(current_time);
      $('.start-time').text(currentTime);

      var progressbar = document.getElementById('seekBar');
      progressbar.value = (player.currentTime / player.duration);

      if (player.currentTime == player.duration) {
        $('#play-button').removeClass('pause');
      }
    }
  }

  function initAudioPlayer() {

    var $player = $('#player');
    $player.bind('timeupdate', initProgressBar);
    var isPlaying = false;

    // Audio Player Controls
    $('#play-button').on('click', togglePlay);
    $('#mute-button').on('click', toggleMute);
    $('#play-button').addClass('pause');

    function togglePlay() {
      if ($player.prop('paused')) {
        $player[0].play();
        $('#play-button').addClass('pause');
        isPlaying = true;
      } else {
        $player[0].pause();
        $('#play-button').removeClass('pause');
        isPlaying = false;
      }
    }

    function toggleMute() {
      if ($player.prop('volume') == 0) {
        $player.prop('volume', '1.0');
        $('#mute-button').removeClass('muted');
      } else {
        $player.prop('volume', '0');
        $('#mute-button').addClass('muted');
      }
    }

    // Click-To-Seek Functionality
    var $progressbar = $('#seekBar');
    $progressbar.on('click', seek);

    function seek(evt) {
      var percent = evt.offsetX / this.offsetWidth;
      $player.prop('currentTime', (percent * $player.prop('duration')));
      $progressbar.prop('value', (percent / 100));
    }
  }



  // Speaker Rows
  var $speakerRows = $('.speaker-row');
  $speakerRows.on('click', function (event) {

    var prevSelectedSpeaker = $('.speaker-row-selected').find('h3').text();
    var selectedSpeaker = $(this).find('h3').text();

    if (selectedSpeaker != prevSelectedSpeaker) {
      $speakerRows.removeClass('speaker-row-selected');
      $(this).addClass('speaker-row-selected');

      // Display appropriate style menu items
      renderStyleMenu(selectedSpeaker);

      // Dismiss modal
      if ($modal.is(':visible') && $(event.target).attr('class') != 'demo-link') {
        closeModal();
      }
    }
  });

  var $demoLinks = $('.demo-link');
  $demoLinks.on('click', function (event) {
    var sample_files = {
      'demo-alicia': ['Alicia', 'demo-alicia.mp3'],
      'demo-hilary': ['Hilary', 'demo-hilary.mp3'],
      'demo-liam': ['Liam', 'demo-liam.mp3']
    };
    closeModal();
    displayModal(sample_files[this.id][0], sample_files[this.id][1]);
  });



  // Character counter
  var $mainTextArea = $('#text');
  var $charCounter = $('#character-counter');
  var maxLength = 300;

  $mainTextArea.on('keyup', function (event) {
    clearMessage();

    var currentLength = $mainTextArea.val().length;
    $charCounter.text(currentLength + '/' + maxLength);

    if (currentLength > maxLength) {
      displayError(true);
    } else {
      clearError(true);
    }
  });

  $mainTextArea.on('keydown', function (event) {
    closeModal();
  });

  $mainTextArea.on('focus', function (event) {
    enableListenButton();
  });



  // Style Menu
  var speakersAndStyles = {
    'Alicia': [
      ['Conversational', 'conversational'],
      ['Narration', 'narration'],
      ['Happy', 'happy'],
      ['Sad', 'sad'],
      ['Scared', 'scared'],
      ['Angry', 'angry'],
      ['Flirty', 'flirty'],
      ['Flustered', 'flustered'],
      ['Whispering', 'whispering']
    ],
    'Liam': [
      ['Conversational', 'conversational'],
      ['Narration', 'narration'],
      ['Sad', 'sad'],
      ['Scared', 'scared'],
      ['Angry', 'angry'],
      ['Flirty', 'flirty'],
      ['Flustered', 'flustered'],
      ['New Yorker', 'new-yorker'],
      ['Robot', 'robot']
    ],
    'Hilary': [
      ['Conversational', 'conversational'],
      ['Narration', 'narration'],
      ['Happy', 'happy'],
      ['Sad', 'sad'],
      ['Scared', 'scared'],
      ['Angry', 'angry'],
      ['Flirty', 'flirty'],
      ['Flustered', 'flustered'],
      ['Whispering', 'whispering']
    ]
  };

  function renderStyleMenu(speakerName) {
    // Creates style menu with appropriate menu items for selected speaker.

    var styles = speakersAndStyles[speakerName];

    var $styleMenu = $('<div>', {
      class: 'style-menu-container',
    });

    $.each(styles, function (i, style) {

      var $styleMenuItem = $('<div>', {
        class: 'style-menu-item',
        'data-style': style[1],
      });

      var $styleMenuItemImg = $('<img>', {
        src: ('../static/images/' + style[1] + '@2x.png')
      });

      $styleMenuItem.on('click', styleMenuItemClicked);
      $styleMenuItem.append($styleMenuItemImg);
      $styleMenu.append($styleMenuItem);
    });

    $('.style-menu-container').replaceWith($styleMenu);

    // select first menu item as default style
    var $firstMenuItem = $('.style-menu-item').first();
    var selectedStyle = $firstMenuItem.attr('data-style');
    $firstMenuItem.addClass('selected-style');
    $firstMenuItem.find('img').attr('src', ('/static/images/' + selectedStyle + '_selected@2x.png'));
  }

  function styleMenuItemClicked() {
    if (!$(this).hasClass('selected-style')) {
      closeModal();

      // update thumbnail to selected state
      $(this).find('img').attr('src', ('/static/images/' + $(this).attr('data-style') + '_selected@2x.png'));

      // deselect previously-selected item
      $prevSelectedStyle = $('.selected-style');
      $prevSelectedStyle.find('img').attr('src', ('/static/images/' + $prevSelectedStyle.attr('data-style') + '@2x.png'));
      $prevSelectedStyle.removeClass('selected-style');

      $(this).addClass('selected-style');
    }
  }



  // Display messages
  function displayError(charCountError) {
    $mainTextArea.prop('disabled', false);
    $mainTextArea.css('margin', '0px');
    $mainTextArea.css('border', '2px solid #FF3366');
    if (charCountError) {
      $charCounter.css('color', '#FF3366');
    }
  }

  function clearError(charCountError) {
    $mainTextArea.css('margin', '1px');
    $mainTextArea.css('border', '1px solid #D9E1F1');
    if (charCountError) {
      $charCounter.css('color', '#C7D1E2');
    }
  }

  function displayMessage(message) {
    var $errorMessage = $('.error');
    $errorMessage.hide();
    $errorMessage.html('<p>' + message + '</p>');
    $errorMessage.show();
  }

  function clearMessage() {
    $('.error').html('');
  }



  // Listen button
  function toggleListenButtonLoading() {
    if ($listenButton.prop('disabled')) {
      enableListenButton();
    } else {
      $listenButton.prop('disabled', true);
      $listenButton.text('Loading');
      $listenButton
        .removeClass('btn-enabled')
        .addClass('btn-disabled');
    }
  }

  $listenButton.on('click', function (event) {
    clearMessage();
    closeModal();
    $mainTextArea.prop('disabled', true);

    var text = $mainTextArea.val();
    var speaker = $('.speaker-row-selected').find('h3').text();
    var style = $('.selected-style').attr('data-style');
    var requestData = {
      text: text,
      speaker: speaker,
      style: style
    };

    if (text.length == 0) {
      displayError(true);
      return;
    } else if (text.length > maxLength) {
      displayError();
      return;
    }

    toggleListenButtonLoading();

    $.ajax({
      type: 'POST',
      contentType: 'application/json; charset=utf-8',
      url: 'synthesize',
      data: JSON.stringify(requestData),
      success: function (data) {
        disableListenButton();
        $mainTextArea.prop('disabled', false);
        if (data['filename']) {
          displayModal(speaker, data['filename']);
        } else {
          displayMessage('Unable to retrieve audio file. Please try again!')
        }
      },
      error: function (xhr, status, error) {
        // TODO: add error details in message shown to user
        $mainTextArea.prop('disabled', false);
        displayMessage('Something went wrong! Please try again.');
        enableListenButton();
      },
      dataType: 'json'
    });

    ga('send', {
      hitType: 'event',
      eventCategory: 'Listen',
      eventAction: speaker,
      eventLabel: text,
    });
  });



  $modal.hide();
  $closeButton.hide();
  initAudioPlayer();
  renderStyleMenu($('.speaker-row-selected').find('h3').text());
});
