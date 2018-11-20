$( document ).ready( function() {

  // Modal
  var $modal = $( "#modal" );
  var $closeButton = $( ".close" );
  var $listenButton = $( "#listen-button" );

  function displayModal( speaker, filename ) {
    $modal.show();
    $closeButton.fadeIn( 3000 );

    disableListenButton();

    // Update speaker row
    var $avatar = $( ".modal .speaker-row-modal img" );
    $avatar.attr( "src", "static/images/avatar_" + speaker.toLowerCase() + ".png" );
    $( ".modal .speaker-details-modal h3" ).text( speaker );

    // Update audio file
    var $audioPlayer = $( "#player" );
    var $audioSource = $( "#player source" );
    $audioSource.attr( "src", "../static/samples/" + filename );
    $audioPlayer[ 0 ].pause();
    $audioPlayer[ 0 ].load();

    // Update play button state
    $( "#play-button" ).addClass( "pause" );

    // Update download link
    $( "#download-link" ).attr( "href", "../static/samples/" + filename + "?attachment=True" );
  }

  function closeModal() {
    var $audioPlayer = $( "#player" );
    $audioPlayer[ 0 ].pause();
    $audioPlayer[ 0 ].currentTime = 0;
    $modal.hide();
    $closeButton.hide();
    enableListenButton();
  }

  $closeButton.on( "click", function() {
    closeModal();
  } );

  $( ".switch" ).on( "click", enableListenButton );

  function disableListenButton() {
    $listenButtonProgressBar.css( "width", "100%" );
    $listenButton.prop( "disabled", true );
    $listenButton.children( "p" ).text( "Listen" );
    $listenButton
      .removeClass( "btn-enabled" )
      .addClass( "btn-disabled" );
  }

  function enableListenButton() {
    $listenButtonProgressBar.css( "display", "None" );
    $listenButtonProgressBar.css( "width", "0%" );
    $listenButtonProgressBar.css( "display", "block" );
    $listenButton.prop( "disabled", false );
    $listenButton.children( "p" ).text( "Listen" );
    $listenButton
      .removeClass( "btn-disabled" )
      .addClass( "btn-enabled" );
  }

  function calculateTotalValue( length ) {
    var minutes = Math.floor( length / 60 );
    var secondsInt = length - minutes * 60;
    var seconds = Math.round( secondsInt );

    seconds = ( seconds < 10 ? "0" + seconds : seconds );
    var time = ( minutes < 1 ? ( "00:" + seconds ) : ( minutes + ":" + seconds ) );

    return time;
  }

  function calculateCurrentValue( currentTime ) {
    var currentMinute = parseInt( currentTime / 60 ) % 60,
      currentSecondsLong = currentTime % 60,
      currentSeconds = currentSecondsLong.toFixed(),
      currentTime =
      ( currentMinute < 1 ? "0" + currentMinute : currentMinute ) +
      ":" +
      ( currentSeconds < 10 ? "0" + currentSeconds : currentSeconds );

    return currentTime;
  }

  function initProgressBar() {
    var player = document.getElementById( "player" );
    var length = player.duration;
    var currentTime = player.currentTime;

    if ( !isNaN( length ) ) {
      var totalLength = calculateTotalValue( length );
      $( ".end-time" ).text( totalLength );

      var currentTime = calculateCurrentValue( currentTime );
      $( ".start-time" ).text( currentTime );

      var progressbar = document.getElementById( "seekBar" );
      progressbar.value = ( player.currentTime / player.duration );

      if ( player.currentTime === player.duration ) {
        $( "#play-button" ).removeClass( "pause" );
      }
    }
  }

  function initAudioPlayer() {

    var $player = $( "#player" );
    $player.bind( "timeupdate", initProgressBar );

    // Audio Player Controls
    $( "#play-button" ).on( "click", togglePlay );
    $( "#mute-button" ).on( "click", toggleMute );
    $( "#play-button" ).addClass( "pause" );

    function togglePlay() {
      if ( $player.prop( "paused" ) ) {
        $player[ 0 ].play();
        $( "#play-button" ).addClass( "pause" );
      } else {
        $player[ 0 ].pause();
        $( "#play-button" ).removeClass( "pause" );
      }
    }

    function toggleMute() {
      if ( $player.prop( "volume" ) === 0 ) {
        $player.prop( "volume", "1.0" );
        $( "#mute-button" ).removeClass( "muted" );
      } else {
        $player.prop( "volume", "0" );
        $( "#mute-button" ).addClass( "muted" );
      }
    }

    // Click-To-Seek Functionality
    var $progressbar = $( "#seekBar" );
    $progressbar.on( "click", seek );

    function seek( evt ) {
      var percent = evt.offsetX / this.offsetWidth;
      $player.prop( "currentTime", ( percent * $player.prop( "duration" ) ) );
      $progressbar.prop( "value", ( percent / 100 ) );
    }
  }


  // Speaker Rows
  var $speakerRows = $( ".speaker-row" );
  $speakerRows.on( "click", function( event ) {

    var prevSelectedSpeaker = $( ".speaker-row-selected" ).find( "h3" ).text();
    var selectedSpeaker = $( this ).find( "h3" ).text();

    if ( selectedSpeaker !== prevSelectedSpeaker ) {
      $speakerRows.removeClass( "speaker-row-selected" );
      $( this ).addClass( "speaker-row-selected" );

      // Dismiss modal
      if ( $modal.is( ":visible" ) && $( event.target ).attr( "class" ) !== "demo-link" ) {
        closeModal();
      }
    }
  } );

  var $demoLinks = $( ".demo-link" );
  $demoLinks.on( "click", function() {
    var sampleFiles = {
      "demo-linda": [ "Linda", "demo-linda.mp3" ]
    };
    closeModal();
    displayModal( sampleFiles[ this.id ][ 0 ], sampleFiles[ this.id ][ 1 ] );
  } );


  // Character counter
  var $mainTextArea = $( "#text" );
  var $charCounter = $( "#character-counter" );
  var maxLength = 300;

  $mainTextArea.on( "keyup", function() {
    clearMessage();

    var currentLength = $mainTextArea.val().length;
    $charCounter.text( currentLength + "/" + maxLength );

    if ( currentLength > maxLength ) {
      displayError( true );
    } else {
      clearError( true );
    }
  } );

  $mainTextArea.on( "keydown", function() {
    closeModal();
  } );

  $mainTextArea.on( "focus", function() {
    enableListenButton();
  } );


  // Display messages
  function displayError( charCountError ) {
    $mainTextArea.prop( "disabled", false );
    $mainTextArea.css( "margin", "0px" );
    $mainTextArea.css( "border", "2px solid #FF3366" );
    if ( charCountError ) {
      $charCounter.css( "color", "#FF3366" );
    }
  }

  function clearError( charCountError ) {
    $mainTextArea.css( "margin", "1px" );
    $mainTextArea.css( "border", "1px solid #D9E1F1" );
    if ( charCountError ) {
      $charCounter.css( "color", "#C7D1E2" );
    }
  }

  function displayMessage( message ) {
    var $errorMessage = $( ".text-editor-error" );
    $errorMessage.hide();
    $errorMessage.html( "<p>" + message + "</p>" );
    $errorMessage.show();
  }

  function clearMessage() {
    $( ".text-editor-error" ).html( "" );
  }


  // Listen button
  function toggleListenButtonLoading() {
    if ( $listenButton.prop( "disabled" ) ) {
      enableListenButton();
    } else {
      $listenButton.prop( "disabled", true );
      $listenButton.children( "p" ).text( "Loading" );
      $listenButton
        .removeClass( "btn-enabled" )
        .addClass( "btn-disabled" );
    }
  }

  var listenButtonProgressInterval;
  var $listenButtonProgressBar = $( "#listen-button .btn-progress-bar" );


  function listenButtonProgress( totalTime ) {
    var width = 0;

    listenButtonProgressInterval = setInterval( function() {
      width += 1;
      $listenButtonProgressBar.css( "width", width + "%" );
      if ( width >= 95 ) {
        clearInterval( listenButtonProgressInterval );
      }
    }, totalTime / 100 );
  }

  $listenButton.on( "click", function() {
    clearMessage();
    closeModal();
    $mainTextArea.prop( "disabled", true );

    var text = $mainTextArea.val();
    var speaker = $( ".speaker-row-selected" ).find( "h3" ).text();
    var isHighFidelity = $( ".text-editor-footer .switch input" ).is( ":checked" );
    var requestData = {
      text: text.trim(),
      speaker: speaker,
      isHighFidelity: isHighFidelity
    };

    if ( text.length === 0 ) {
      displayError( true );
      return;
    } else if ( text.length > maxLength ) {
      displayError();
      return;
    }

    toggleListenButtonLoading();


    // Estimate of 300 frames a second with 6 frames per character
    var estimatedSpectrogramModelProcessing = text.length * 6 / 300;

    // Add an extra second for any other operations.
    var estimatedLag = 1000;

    // TODO: Parameterize this and document
    var estimatedTime = estimatedLag + estimatedSpectrogramModelProcessing;
    if ( isHighFidelity ) {

      // 6 frames per character with 12.5ms for each frame
      var estimatedSampleLength = text.length * 6 * 12.5;

      // Assumes the vocoder generates at 1100 samples per second
      var estimatedVocoderProcessing = ( 24000 / 1100 ) * estimatedSampleLength;
      estimatedTime += ( estimatedVocoderProcessing +
        estimatedSpectrogramModelProcessing +
        estimatedLag );
    } else {
      var estimatedGriffinLimProcessing = 2500;
      estimatedTime = ( estimatedGriffinLimProcessing +
        estimatedSpectrogramModelProcessing +
        estimatedLag );
    }

    listenButtonProgress( estimatedTime );

    $.ajax( {
      type: "POST",
      contentType: "application/json; charset=utf-8",
      url: "synthesize",
      data: JSON.stringify( requestData ),
      success: function( data ) {
        disableListenButton();
        $mainTextArea.prop( "disabled", false );
        if ( data.filename ) {
          displayModal( speaker, data.filename );
        } else {
          displayMessage( "Unable to retrieve audio file. Please try again!" );
        }
        clearInterval( listenButtonProgressInterval );
      },
      error: function() {
        clearInterval( listenButtonProgressInterval );

        // TODO: add error details in message shown to user
        $mainTextArea.prop( "disabled", false );
        displayMessage( "Something went wrong, sorry :( Sometimes this demo fails with numerical " +
          "characters and none-standard characters. If all else fails, try refreshing and " +
          "emailing <a href='michaelp@allenai.org'>michaelp@allenai.org</a>." );
        enableListenButton();
      },
      dataType: "json"
    } );
  } );


  $modal.hide();
  $closeButton.hide();
  initAudioPlayer();
} );