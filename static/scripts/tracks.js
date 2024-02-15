/**
 * Filename: tracks.js
 * Author: Vivian Liu
 * Functions here manage logic that is specific to the tracks area of the application.
 */




/**
 * Processes data with a callback function.
 *
 * @param {any} No params but collects data from the form to pass to the backend as a JSON payload.
 * @returns {any} None but triggers the tracks container to update to reflect the newly generated interval.
 */
function generateInterval() {

    console.log($("#audioReactivityFlag")[0].checked);
  
    $('#loading').show();
  
  json_data = JSON.stringify(localStorage.regions)

  $.ajax({
      type: "POST",
      url: '/generate_interval',
      timeout: 1000 * 60 * 3,
      data: JSON.stringify({ 
      "json_data":json_data,
      "current_interval_start": $("#start")[0].value,
      "current_interval_end": $("#end")[0].value,
      "interval_num": $("#interval_num")[0].value,

      "start_note": $("#start_note")[0].value,
      "end_note": $("#end_note")[0].value,
      "start_seed": $("#start_seed")[0].value,
      "end_seed": $("#end_seed")[0].value,
      "audio_reactive_flag": $("#audioReactivityFlag")[0].checked,

      // "end_interval_note": $("#endnote")[0].value,
      } ),
      processData: false,
      cache: false,
      async: true,
      contentType: 'application/json;charset=UTF-8',
      success: function(data) {
        $('#loading').hide();
        updateTracks();

        
         
          
     },
      error: function (request, status, error) {
          clearInterval(handle);
          console.log("Error");

      }
  });
}

/**
 * Processes data with a callback function.
 *
 * @param {any} No params but collects start prompt and seed from the form to pass to the backend as a JSON payload.
 * @returns {any} None but triggers the generations container to update to reflect the newly generated image.
 */
function generateBrainstormImg() {

  $("#loading").show();
  var relevantData = {
    "start_prompt": $("#start_prompt")[0].value,
    "test_seed": $("#test_seed")[0].value,
    // "end_prompt": $("#end_prompt")[0].value,
    
  }

  $.ajax({
    type: "POST",
    url: '/generate_brainstorm_img',
    data: JSON.stringify(relevantData)
    ,
    processData: false,
    cache: false,
    async: true,
    contentType: 'application/json;charset=UTF-8',
    success: function(data) {
      $("#loading").hide();
      $("#generations_container").load( " #generations_container");
    
   },
    error: function (request, status, error) {
        clearInterval(handle);
        console.log("Error");

    }
  });

}


let added_frames = 0;
function addFrame(progress) {

    let left_margin = screen.width * progress;
    
    let idd="frame" + added_frames.toString();
    
    $("#video_track").append("<div id='" + idd + "' class='frame_preview'></div>");
    $("#" +idd).css( 'position', 'absolute' );

    $("#" +idd).css( 'margin-left', left_margin );
    // $("#video_track")
    //  + "left='" + left_margin.toString() + "px'>";
    added_frames += 1;
}


function generateEndImage(prompt) {

  $('#loading').show();

  json_data = JSON.stringify(localStorage.regions)

  $.ajax({
    type: "POST",
    url: '/call_generate_end',
    data: JSON.stringify({ 
    "prompt": $("#end_note")[0].value,
    "seed":$("#end_seed")[0].value,
      "interval_num":  $("#interval_num")[0].value,
    "json_data":json_data,
    } ),
    processData: false,
    cache: false,
    async: false,
    contentType: 'application/json;charset=UTF-8',
    success: function(data) {
      $('#loading').hide();

      $("#end_frame_preview")[0].src = "./static/previews/" + $("#interval_num")[0].value.toString() + "_end.jpg";
 
        
   },
    error: function (request, status, error) {
        clearInterval(handle);
        console.log("Error");

    }
});

$.ajax({
  type: "POST",
  url: '/call_generate_end',
  data: JSON.stringify({ 
  "prompt": $("#end_note")[0].value,
  "seed":$("#end_seed")[0].value+1,
    "interval_num":  $("#interval_num")[0].value,
  "json_data":json_data,
  } ),
  processData: false,
  cache: false,
  async: false,
  contentType: 'application/json;charset=UTF-8',
  success: function(data) {
    $('#loading').hide();

    $("#end_frame_preview")[0].src = "./static/previews/" + $("#interval_num")[0].value.toString() + "_end.jpg";

      
 },
  error: function (request, status, error) {
      clearInterval(handle);
      console.log("Error");

  }
});
}

$('#loading').bind('ajaxStart', function(){
  alert("sdfd");
  $(this).show();
}).bind('ajaxStop', function(){
  $(this).hide();
});


function generateImage(prompt) {

  $('#loading').show();

  json_data = JSON.stringify(localStorage.regions);

  //getting form into Jquery Wrapper Instance to enable JQuery Functions on form                    
  var form = $("#full-form");

  //Serializing all For Input Values (not files!) in an Array Collection so that we can iterate this collection later.
  var params = form.serializeArray();

  //Getting Files Collection
  var files = $("#initialImg")[0].files;

  //Declaring new Form Data Instance  
  var formData = new FormData();

  //Looping through uploaded files collection in case there is a Multi File Upload. This also works for single i.e simply remove MULTIPLE attribute from file control in HTML.  
  for (var i = 0; i < files.length; i++) {
      formData.append(files[i].name, files[i]);
  }
  //Now Looping the parameters for all form input fields and assigning them as Name Value pairs. 
  $(params).each(function (index, element) {
      formData.append(element.name, element.value);
  });

  formData.append("imgChecked", $("#imgChecked")[0].checked);
  console.log(formData);


  $.ajax({
    type: "POST",
    url: '/call_generate',
    data: formData

    ,
    processData: false,
    cache: false,
    async: false,
    contentType: 'application/json;charset=UTF-8',
    success: function(data) {
      $('#loading').hide();

    
   },
    error: function (request, status, error) {
        clearInterval(handle);
        console.log("Error");

    }
  });

}

/**
 * Deletes a stitched video, triggers the /delete_stitched_video endpoint.
 *
 */
function deleteStitchedVideo() {
  $.ajax({
    type: "POST",
    url: '/delete_stitched_video',
    timeout: 1000 * 60 * 3,
    data: JSON.stringify({ 
    
    } ),
    processData: false,
    cache: false,
    async: true,
    contentType: 'application/json;charset=UTF-8',
    success: function(data) {
      
     
      $( ".video_area" ).load(window.location.href + " .video_area" );
      $("#deleteVideo").hide();

      
       
        
   },
    error: function (request, status, error) {
        clearInterval(handle);
        console.log("Error");

    }
});

}


/**
 * Calls generate_video endpoint. Deprecated for generate_interval. TO-DO: delete.
 */
function generateVideo() {

  $('#loading').show();
  json_data = JSON.stringify(localStorage.regions)

  $.ajax({
      type: "POST",
      url: '/generate_video',
      data: JSON.stringify({ 
        // "start_prompt": start_prompt,
      "json_data":json_data,
      // "end_prompt": end_prompt,
      // "seed":seed
        } ),
      processData: false,
      cache: false,
      async: false,
      contentType: 'application/json;charset=UTF-8',
      success: function(data) {
        $('#loading').hide();
         
          
     },
      error: function (request, status, error) {
          clearInterval(handle);
          console.log("Error");

      }
  });
}


/**
 * Refreshes track area to reflect the newly generated track.
 */
function updateTracks()
    { 
        $( "#tracks_container" ).load(window.location.href + " #tracks_container" );
    }


/**
 * Plays the element. 
 *
 */
function playRegionVideo(element) {
    
 
  $(element)[0].play();
    
}

// could get it to delete the numbers around it too
function dropPromptOnly(ev) {
  ev.preventDefault();
  var data = ev.dataTransfer.getData("text");

  ev.target.src = document.getElementById(data).src;
  $("#start_prompt")[0].value = document.getElementById(data).id.toString().slice(0,-4).replaceAll("_"," ").replaceAll("-",",");
  prompt_and_seed = $("#start_prompt")[0].value;
  let re2 = new RegExp("\\d+(.*)\\s\\d+");
    
  $("#start_prompt")[0].value = prompt_and_seed.match(re2)[1];

  let re = new RegExp("(\\d+)",'g');


      const found = prompt_and_seed.matchAll(re);
    
      var seed ;
      for (const match of found) {

        console.log(
          `Found ${match[0]} start=${match.index} end=${
            match.index + match[0].length
          }.`,
        );
        seed = match[0];
      }
      console.log(seed  );

      $("#test_seed")[0].value = seed;
}


/**
 * allowDrop of drag-and-drop functionality, which allows the dropover on an element.
 *
 */
function allowDrop(ev) {
    ev.preventDefault();
  }

/**
* Drag of drag-and-drop functionality. Transfers data attached to element.
*
*/
function drag(ev) {
ev.dataTransfer.setData("text", ev.target.id);

}
  
let found_matches;


/**
 * Drop of drag-and-drop functionality. 
 * Regex which parses the path on the image and then splits it into start prompt and seed. Then it autopopulates it on the form and replaces the image preview.
 *
 */
function drop(ev) {
    ev.preventDefault();

    // this gets generation num 
    var data = ev.dataTransfer.getData("text");
    console.log(ev);
    // ev.target.innerHTML="";
    ev.target.src = document.getElementById(data).src;
    var start_or_end = "";

    if (ev.target.id == "start_frame_preview") {
      start_or_end= "start";
     

      $("#start_note")[0].value = document.getElementById(data).id.toString().slice(0,-4).replaceAll("_"," ").replaceAll("-",",");
      prompt_and_seed = $("#start_note")[0].value;

      let re2 = new RegExp("\\d+(.*)\\s\\d+");
    
      $("#start_note")[0].value = prompt_and_seed.match(re2)[1];


      let re = new RegExp("(\\d+)",'g');


      const found = prompt_and_seed.matchAll(re);
    
      var seed ;
      for (const match of found) {

        console.log(
          `Found ${match[0]} start=${match.index} end=${
            match.index + match[0].length
          }.`,
        );
        seed = match[0];
      }
      console.log(seed  );

      $("#start_seed")[0].value = seed;

      
      // console.log($("#" + document.getElementById(data).id)[0]);
      
    }

    if (ev.target.id == "end_frame_preview") {
      $("#end_note")[0].value = document.getElementById(data).id.toString().slice(0,-4).replaceAll("_"," ").replaceAll("-",",");

      prompt_and_seed = $("#end_note")[0].value;

      console.log($("#end_note")[0].value);
      // const re = new RegExp("(.*)_(\d+.jpg)");
      // console.log($("#" + document.getElementById(data).id)[0]);
      start_or_end= "end"

      let re2 = new RegExp("\\d+(.*)\\s\\d+");
    
      $("#end_note")[0].value = prompt_and_seed.match(re2)[1];


      let re = new RegExp("(\\d+)",'g');


      const found = prompt_and_seed.matchAll(re);
    
      var seed ;
      for (const match of found) {

        console.log(
          `Found ${match[0]} start=${match.index} end=${
            match.index + match[0].length
          }.`,
        );
        seed = match[0];
      }
      console.log(seed  );

      $("#end_seed")[0].value = seed;
      
    }

    
   
    /**
     * Ajax call triggers the image preview to also update to the new image that was dragged and dropped.
     *
     */

    $.ajax({
      type: "POST",
      url: '/replace_preview',
      data: JSON.stringify({ 
        "intervalNum": $("#interval_num")[0].value,
      "draggedPath":ev.target.src,
      "start_or_end":start_or_end
      // "end_prompt": end_prompt,
      // "seed":seed
        } ),
      processData: false,
      cache: false,
      async: false,
      contentType: 'application/json;charset=UTF-8',
      success: function(data) {
        console.log(ev.target.src);
        saveRegions();
        $( "#preview_history_area" ).load(window.location.href + " #preview_history_area" );

         
          
     },
      error: function (request, status, error) {
          clearInterval(handle);
          console.log("Error");

      }
  });
}
