function hideTrack(element) {
    srcToDelete = element.parentNode.children[1].src;
    console.log(element.parentNode.children[1].src);

    $.ajax({
        type: "POST",
        url: '/delete_track',
        data: JSON.stringify({ 

            'path_to_delete': srcToDelete,

        } ),
        processData: false,
        cache: false,
        async: false,
        contentType: 'application/json;charset=UTF-8',
        success: function(data) {
        //    alert("returned")
            updateTracks();
            
            
        },
        error: function (request, status, error) {
            clearInterval(handle);
            console.log("Error");
    
        }
    });

}   
function varyImage(element) {

  $("#start_prompt")[0].value = element.parentNode.children[1].id.slice(0,-4).replaceAll("_"," ").replaceAll("-",",");
  prompt_and_seed = $("#start_prompt")[0].value;
  let re2 = new RegExp("\\d+(.*)\\s\\d+");
    
  $("#start_prompt")[0].value = prompt_and_seed.match(re2)[1];

  let re = new RegExp("(\\d+)",'g');

      const found = prompt_and_seed.matchAll(re);
    
      var seed ;
      for (const match of found) {

        seed = match[0];
      }
      console.log(seed  );

      $("#test_seed")[0].value = seed;

      generateBrainstormImg();
}
    



function hideContainer(element) {
    // console.log(element.parentNode.children[1].src);
    imgSrcDeleted = element.parentNode.children[1].src;
    console.log(element.parentNode);
    
    
    element.parentNode.remove();


    $.ajax({
        type: "POST",
        url: '/delete_file',
        data: JSON.stringify({ 

            'path_to_delete': imgSrcDeleted,

        } ),
        processData: false,
        cache: false,
        async: false,
        contentType: 'application/json;charset=UTF-8',
        success: function(data) {
        //    alert("returned")
            
            
        },
        error: function (request, status, error) {
            clearInterval(handle);
            console.log("Error");
    
        }
    });
}



function sendInitialImg(elem) {

    var file_data = elem.files[0];
    console.log(file_data);
    $("#initialImgPreview")[0].src = URL.createObjectURL(file_data); // set src to blob url
    
    $.ajax({
        type: "POST",
        url: '/receive_img_blob',
        data: JSON.stringify({ 

            img_url: $("#initialImgPreview")[0].src,
            img_data : file_data
        } ),
        processData: false,
        cache: false,
        async: false,
        contentType: 'application/json;charset=UTF-8',
        success: function(data) {
           alert("returned")
            
            
        },
        error: function (request, status, error) {
            clearInterval(handle);
            console.log("Error");
    
        }
    });



    // console.log($("#initialImgPreview")[0].src);
        //   userimage = document.getElementById('myImgSrc');

        //   userimage.addEventListener("load", function () {

        //   var form_data = new FormData();

        //   form_data.append('file', file_data);
        //   form_data.append("starter_image", "true");


    // $("#initialImgPreview")[0].src = document.forms.namedItem.


}


function sendImg2Img() {

    $.ajax({
        type: "POST",
        url: '/generate_img2img',
        data: JSON.stringify({ 



        } ),
        processData: false,
        cache: false,
        async: false,
        contentType: 'application/json;charset=UTF-8',
        success: function(data) {
           alert("returned")
            
            
        },
        error: function (request, status, error) {
            clearInterval(handle);
            console.log("Error");
    
        }
    });
}

function setDefaultImg(elem) {
    elem.src = 'https://northphoenixprep.greatheartsamerica.org/wp-content/uploads/sites/21/2016/11/default-placeholder.png'
    
    // this.src='demo.jpg?'+new Date().getTime();


}

function regenerateFrame(element) {
    promptToRegenerate = element.parentNode.parentNode.children[1].children[0].innerText;
    console.log(element.parentNode.parentNode.children[1].children[0]);

    generateImage(promptToRegenerate);
    
}


function stitchVideo() {

    $('#loading').show();
    // videos = Array.from($("#page_right_container")[0].children);
    // videos = videos.slice(1,videos.length);
    videos = $("#tracks_container video");
    let videoPaths = [];
    
    for (let i= 0; i < videos.length; i++) {
        
        videoPaths.push(videos[i].src);
    }


    
    $.ajax({
        type: "POST",
        url: '/stitch_videos',
        data: JSON.stringify({ 

            'video_paths': videoPaths.toString(),

        } ),
        processData: false,
        cache: false,
        async: true,
        contentType: 'application/json;charset=UTF-8',
        success: function(data) {
            $('#loading').hide();
            

            $( ".video_area" ).load(window.location.href + " .video_area" );
            
            setTimeout(() => {$("#deleteVideo").show();}, 500)
            
            
        },
        error: function (request, status, error) {
            clearInterval(handle);
            console.log("Error");
    
        }
    });
      
}

function brainstorm() {
    $.ajax({
        type: "POST",
        url: '/brainstorm',
        data: JSON.stringify({ 

            'goal': $("#description")[0].value,
           

        } ),
        processData: false,
        cache: false,
        async: false,
        contentType: 'application/json;charset=UTF-8',
        success: function(data) {
        

            color_trio = data["color_trio"].split(",");
            angle_trio = data["angle_trio"].split(",");
            time_trio = data["time_trio"].split(",");
            $("#color_start1")[0].innerHTML = color_trio[0];
            $("#color_start2")[0].innerHTML = color_trio[1];
            $("#color_start3")[0].innerHTML = color_trio[2];
            $("#color_start4")[0].innerHTML = color_trio[3];
            $("#color_start5")[0].innerHTML = color_trio[4];

            $("#angle_start1")[0].innerHTML = angle_trio[0];
            $("#angle_start2")[0].innerHTML = angle_trio[1];
            $("#angle_start3")[0].innerHTML = angle_trio[2];
            $("#angle_start4")[0].innerHTML = angle_trio[3];
            $("#angle_start5")[0].innerHTML = angle_trio[4];

            $("#time_start1")[0].innerHTML = time_trio[0];
            $("#time_start2")[0].innerHTML = time_trio[1];
            $("#time_start3")[0].innerHTML = time_trio[2];
            $("#time_start4")[0].innerHTML = time_trio[3];
            $("#time_start5")[0].innerHTML = time_trio[4];

            $(".empty").removeClass("empty");
            $(".active-pill").removeClass("active-pill");
           
            
            
        },
        error: function (request, status, error) {
            clearInterval(handle);
            console.log("Error");

    
        }
    });

}

function brainstormGPT() {

    if ($("#openaikey")[0].value != "") {
        $("#loading").show();
        $.ajax({
            type: "POST",
            url: '/brainstorm_gpt',
            data: JSON.stringify({ 

                'goal': $("#description")[0].value,
            

            } ),
            processData: false,
            cache: false,
            async: true,
            contentType: 'application/json;charset=UTF-8',
            success: function(data) {
                $("#loading").hide();
            
                // $(".content")[0].innerHTML = ""
                
                $("#startSubject1")[0].innerHTML = data["subject1"];
                $("#startSubject2")[0].innerHTML = data["subject2"];
                $("#startSubject3")[0].innerHTML = data["subject3"];


                $(".subject_empty").removeClass("subject_empty");
                brainstorm();
            
                // $("#color_end1")[0].value = data[0][1];
                // $("#angle_end1")[0].value = data[1][1];
                // $("#time_end1")[0].value = data[2][1];


                // "<img width='24px' src='https://cdn-icons-png.flaticon.com/512/858/858150.png'> Color transition:"  + data[0] + "<br>"
                // $(".content")[0].innerHTML += "<img width='24px' src='https://cdn-icons-png.flaticon.com/512/858/858150.png'>Motion transition:"  + data[1] + "<br>"
                // $(".content")[0].innerHTML += "<img width='24px' src='https://cdn-icons-png.flaticon.com/512/858/858150.png'>Perspective transition:"  + data[2] + "<br>"
                
                // $(".content")[0].innerHTML += "<img width='24px' src='https://cdn-icons-png.flaticon.com/512/858/858150.png'>Lyric suggestion:"  + data[3] + ""
                // // $(".content")[0].innerHTML += "</ul>"
                
                
                
            },
            error: function (request, status, error) {
                $("#loading").hide();
                alert("Error from GPT.");
                // brainstormGPT();
                clearInterval(handle);
                console.log("Error");
                
        
            }
        });
    } else {
        alert("Please enter an OpenAI API key in the field on the top right.");
    }
}



function playWaveSurfer() {
    window.wavesurfer.playPause();
}
function openHistory(element) {
    element.classList.toggle("active");
    var content = $("#preview_history_area")[0];
    if (content.style.display === "block") {
        content.style.display = "none";
    } else {
        content.style.display = "block";
    }
}



function changePromptFocus(elem) {
    if (elem.id == $(".active_prompt")[0].id) {
        return
    }
    else {

        $( "#start_prompt" ).toggleClass( "active_prompt");

        $( "#end_prompt" ).toggleClass( "active_prompt");
    }
   

    
}

function subjectPillClick(elem) {
    console.log(elem);
    $(".active_prompt")[0].value += elem.innerText + ",";
}

function pillClick(elem) {
    console.log(elem);
    $(".active_prompt")[0].value += "," + elem.innerText ;
    $(elem).toggleClass('active-pill');
}