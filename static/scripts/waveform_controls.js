/**
 * Controls waveform. Largely wavesurfer.js code.
 *
 */

let ws = window.wavesurfer;

let interval_counter = 0;

const colors = ['rgba(128,128,128,0.2)'];

var GLOBAL_ACTIONS = { // eslint-disable-line
    play: function() {
        window.wavesurfer.playPause();
    },


};

// Bind actions to buttons and keypresses
document.addEventListener('DOMContentLoaded', function() {
    document.addEventListener('keydown', function(e) {


        let map = {
            112: 'play', // f1

        };
        let action = map[e.keyCode];
        if (action in GLOBAL_ACTIONS) {
            if (document == e.target || document.body == e.target || e.target.attributes["data-action"]) {
                e.preventDefault();
            }
            GLOBAL_ACTIONS[action](e);
        }

        if (e.code == 32) {
            e.preventDefault();
            e.stopPropagation();
        }

    });

    // [].forEach.call(document.querySelectorAll('[data-action]'), function(el) {
    //     el.addEventListener('play_music', function(e) {
    //         let action = e.currentTarget.dataset.action;
    //         if (action in GLOBAL_ACTIONS) {
    //             e.preventDefault();
    //             GLOBAL_ACTIONS[action](e);
    //         }
    //     });
    // });
});

// Misc
document.addEventListener('DOMContentLoaded', function() {
    // Web Audio not supported
    if (!window.AudioContext && !window.webkitAudioContext) {
        let demo = document.querySelector('#demo');
        if (demo) {
            demo.innerHTML = '<img src="/example/screenshot.png" />';
        }
    }

    // Navbar links
    let ul = document.querySelector('.nav-pills');
    if ( !ul ) {
        return;
    }

    let pills = ul.querySelectorAll('li');
    let active = pills[0];
    if (location.search) {
        let first = location.search.split('&')[0];
        let link = ul.querySelector('a[href="' + first + '"]');
        if (link) {
            active = link.parentNode;
        }
    }
    active && active.classList.add('active');
});

// Create an instance
var wavesurfer;

// Init & load audio file
document.addEventListener('DOMContentLoaded', function() {
    // Init

    // wavesurfer = WaveSurfer.create({
    //     container: '#waveform',
    //     waveColor: '#4F4A85',
    //     progressColor: '#383351',
    //     url: $("#music")[0].innerText,
    //   })

    // Create wavesurfer element. Url field is important as that controls what music is being loaded. 
    wavesurfer = WaveSurfer.create({
        container: document.querySelector('#waveform'),
        height: 100,
        pixelRatio: 1,
        minPxPerSec: 100,
        scrollParent: true,
        normalize: true,
        splitChannels: false,
        barWidth: 3,
        barRadius: 3,
        cursorWidth: 1,
        responsive:true,
        height: 200,
        barGap: 3,
        url: $("#music")[0].innerText,
        plugins: [

            WaveSurfer.regions.create({
   
                regions: [

                ],
                dragSelection: {
                    slop: 5
                }
            }),

            WaveSurfer.timeline.create({
                container: '#wave-timeline'
            }),
            WaveSurfer.cursor.create()
        ]
    });

    // // Load audio from existing media element
    // let mediaElt = $("#input_video")[0];


    wavesurfer.on('error', function(e) {
        console.warn(e);
    });

    // wavesurfer.load(mediaElt);
    // Default: load clairdelune.wav
    wavesurfer.load("./static/audio/clairdelune.wav");

    wavesurfer.on('ready', function() {
    
        // Apply the gray color to any interval that is drawn
        
        wavesurfer.enableDragSelection({
            color: 'rgba(128,128,128,0.2)'
        });

        // Loads intervals from interval_data, which is being written to by the backend
        wavesurfer.util
            .fetchFile({
                responseType: 'json',
                url: './static/interval_data.json'
            })
            .on('success', function(data) {
                loadRegions(data);
                saveRegions();
            });
    });

    // Triggers the play and pause of a specific interval. Stops on exit of the region.
    wavesurfer.on('region-play', function(region) {
        console.log("playing");
        wavesurfer.play(region.start);
        region.once('out', function() {
            wavesurfer.pause();
        });
    });

    
    // On region click, play the region. Highlight it green to make it visually active. Unset the rest of the intervals to be gray.
    wavesurfer.on('region-click', function(region, e) {
        
        
        unsetColors();
        region.color = 'rgba(0,200,100,0.33)';
        
        editAnnotation(region);
        saveRegions();
        //e.stopPropagation();
        // Play on click, loop on shift click
        e.shiftKey ? region.play: region.play();
        
        // const playEvent = new Event("region-play");
        // Listen for the event.


        // document.dispatchEvent(playEvent);
    });
    wavesurfer.on('region-click', editAnnotation);
    // wavesurfer.on('region-update-end', saveRegions);
    // wavesurfer.on('region-updated', saveRegions);
    // wavesurfer.on('region-removed', saveRegions);
    wavesurfer.on('region-in', showNote);
    wavesurfer.on('region-out', hideNote);
    wavesurfer.setVolume(0.15);

    // $(".lyric").css("position","fixed");
    // $(".lyric").css("left",)

    

    /* Toggle play/pause buttons. */
    let playButton = document.querySelector('#play');
    let pauseButton = document.querySelector('#pause');
    wavesurfer.on('play', function() {
        playButton.style.display = 'none';
        pauseButton.style.display = 'block';
    });
    wavesurfer.on('pause', function() {
        playButton.style.display = 'block';
        pauseButton.style.display = 'none';
    });
});

/**
 * Function that unsets the inactive intervals to be gray.
 *
 */
function unsetColors() {
  Object.keys(wavesurfer.regions.list).forEach(function(id) {
    var region = wavesurfer.regions.list[id];

  
      // Change all other regions' colors to gray
      region.color = 'rgba(128, 128, 128, 0.2)'; // Gray color
    

  //   // Apply the color change visually
    region.updateRender();
  });
}

/**
 * Save annotations to localStorage.
 */
function saveRegions() {

    console.log("saved regions called");

    interval_counter = 0;
    let form = document.forms.edit;
    let region = wavesurfer.regions.list[form.dataset.region];
    

    region.update({
        start: form.elements.start.value,
        end: form.elements.end.value,
        data: {
            start_note: form.elements.start_note.value,
            end_note: form.elements.end_note.value,
            interval_num: form.elements.interval_num.value,
            start_img_src: form.elements.interval_num.value,
            start_img_src: $("#start_frame_preview")[0].src,
            start_seed: form.elements.start_seed.value,

            description: form.elements.description.value,

            end_img_src: $("#end_frame_preview")[0].src,
            end_seed:  form.elements.end_seed.value,

        }

    });


    localStorage.regions = JSON.stringify(
        Object.keys(wavesurfer.regions.list).map(function(id) {
            interval_counter++;
            let region = wavesurfer.regions.list[id];
             return {
                start: region.start,
                end: region.end,
                attributes: region.attributes,
                data: region.data
            };
        })
    );

    json_data = JSON.stringify(localStorage.regions)

    // Ajax call which writes the interval data (local Storage) into JSON in the backend.
    $.ajax({
        type: "POST",
        url: '/save_regions',
        data: JSON.stringify({ 
        
        "json_data":json_data,
        } ),
        processData: false,
        cache: false,
        async: false,
        contentType: 'application/json;charset=UTF-8',
        success: function(data) {

            // region.update({
            //     start: form.elements.start.value,
            //     end: form.elements.end.value,
            //     data: {
            //         start_note: form.elements.start_note.value,
            //         end_note: form.elements.end_note.value,
            //         interval_num: form.elements.interval_num.value,
            //         start_img_src: form.elements.interval_num.value,
            //         start_img_src: $("#start_frame_preview")[0].src,
            //         start_seed: form.elements.start_seed.value,
    
            //         end_img_src: $("#end_frame_preview")[0].src,
            //         end_seed:  form.elements.end_seed.value,
    
            //     }
    
            // });

    },
        error: function (request, status, error) {
            clearInterval(handle);
            console.log("Error");

        }
});

}

/**
 * Binds the save to keypresses. Not exposed to the user, so consider deleting.
 *
 */
$(document).ready(function() {
    $(window).keydown(function(event){
      if(event.keyCode == 13) {
        event.preventDefault();
        console.log("hit entered");
        saveRegions();
        console.log("returned from entered");
        return false;
      }

    //   IF TAB PRESSED

      if(event.keyCode == 9) {
        saveRegions();
      }
    });
  });


/**
 * Load regions from localStorage.
 */
function loadRegions(regions) {
    regions.forEach(function(region) {

        region.color = colors[interval_counter % colors.length];
       
        wavesurfer.addRegion(region);
        interval_counter++;
        

        $("#start_frame_preview")[0].src = "./static/previews/" + region.data["interval_num"] + '_start.jpg';
        $("#end_frame_preview")[0].src = "./static/previews/" + region.data["interval_num"] + '_end.jpg';
        
        

    });
}



/**
 * Random RGBA color.
 */
function randomColor(alpha) {
    
    return (
        'rgba(' +
        [
            ~~(Math.random() * 255),
            ~~(Math.random() * 255),
            ~~(Math.random() * 255),
            alpha || 1
        ] +
        ')'
    );

}

/**
 * Renumbers intervals on the waveform. (Edits the form data)
 *
 */
function renumberIntervals() {
    let form = document.forms.edit;
    let i =1;

    for (let region in wavesurfer.regions.list) {
        console.log(wavesurfer.regions.list[region]["data"]["interval_num"]);
        wavesurfer.regions.list[region]["data"]["interval_num"] = i;
        form.elements.interval_num = i;
        i++;
    }

}

/**
 * Edit annotation for a region.
 */
function editAnnotation(region) {
    // showNote(region);
    console.log(region);

    let form = document.forms.edit;
    form.style.opacity = 1;
    (form.elements.start.value = Math.round(region.start * 100) / 100),
    (form.elements.end.value = Math.round(region.end * 100) / 100);
    form.elements.start_note.value = region.data.start_note || '';
    form.elements.interval_num.value = region.data.interval_num || '';
    form.elements.end_note.value = region.data.end_note || '';

    form.elements.description.value = region.data.description || '';

    // form.elements.start_seed.value = Math.round(Math.random()*100);
    form.elements.end_seed.value = region.data.end_seed || '';
    form.elements.start_seed.value = region.data.start_seed || '';

    form.elements.start_frame_preview.src = "./static/previews/" + form.elements.interval_num.value.toString() + '_start' +'.jpg';

    form.elements.start_frame_preview.nextElementSibling.src = "./static/previews/" + form.elements.interval_num.value.toString() + '_start' +'.jpg';


    form.elements.end_frame_preview.src = "./static/previews/" + form.elements.interval_num.value.toString() + '_end' + '.jpg';

    form.elements.end_frame_preview.nextElementSibling.src = "./static/previews/" + form.elements.interval_num.value.toString() + '_end' +'.jpg';

    form.onsubmit = function(e) {
        e.preventDefault();
        region.update({
            start: form.elements.start.value,
            end: form.elements.end.value,
            data: {
                start_note: form.elements.start_note.value,
                end_note: form.elements.end_note.value,
                interval_num: form.elements.interval_num.value,
                start_img_src: form.elements.interval_num.value,
                start_img_src: $("#start_frame_preview")[0].src,
                start_seed: form.elements.start_seed.value,
                description: form.elements.description.value,

                end_img_src: $("#end_frame_preview")[0].src,
                end_seed:  form.elements.end_seed.value,

            }

        });
        e.stopPropagation();
        // form.style.opacity = 0;
    };
    form.onreset = function() {
        // saveRegions();
        e.stopPropagation();
        form.elements.start_img_src.value = form.elements.interval_num.value + 'start.jpg';
        form.elements.end_img_src.value = form.elements.interval_num.value + 'end.jpg';

        // form.style.opacity = 0;
        // form.dataset.region = null;
    };
    form.dataset.region = region.id;

    renumberIntervals();
}

/**
 * Bind controls.
 */
// window.GLOBAL_ACTIONS['delete-region'] = function() {
//     let form = document.forms.edit;
//     let regionId = form.dataset.region;
//     if (regionId) {
//         wavesurfer.regions.list[regionId].remove();
//         form.reset();
//     }
// };

/**
 * Deletes a region in the intervals 
 */
function deleteRegion() {

    let form = document.forms.edit;
    let regionId = form.dataset.region;

    

    // console.log($("#interval_num")[0].value);
    if (regionId) {
        wavesurfer.regions.list[regionId].remove();
        form.reset();
    }

    let i =1;

    // wavesurfer.regions.list references the intervals
    for (let region in wavesurfer.regions.list) {
        console.log(wavesurfer.regions.list[region]["data"]["interval_num"]);
        wavesurfer.regions.list[region]["data"]["interval_num"] = i;
        form.elements.interval_num = i;
        i++;
    }
    saveRegions();

}

// window.GLOBAL_ACTIONS['export'] = function() {
//     window.open(
//         'data:application/json;charset=utf-8,' +
//             encodeURIComponent(localStorage.regions)
//     );
// };

/**
 * Shows data bound to an interval. Unused and consider deleting, but kept for reference. From wavesurfer.js example.
 *
 */
function showNote(region) {
    // console.log("showing");
    // if (!showNote.el) {
    //     showNote.el = document.querySelector('#subtitle');
    // }
    // showNote.el.style.color = 'Red';
    // showNote.el.style.fontSize = 'large';
    // showNote.el.textContent =  " 🎵" + region.data.start_note + "🎵" || '–';
}

/**
 * Hides data bound to an interval. From wavesurfer.js example.
 *
 */

function hideNote(region) {
    if (!hideNote.el) {
        hideNote.el = document.querySelector('#subtitle');
    }
    hideNote.el.style.color = 'Red';
    hideNote.el.style.fontSize = 'large';
    hideNote.el.textContent = '–';
}

/**
 * Sorts dictionary by their start time
 *
 */
// from https://stackoverflow.com/questions/25500316/sort-a-dictionary-by-value-in-javascript
function sort_object(obj) {
    items = Object.keys(obj).map(function(key) {
        return [key, obj[key]];
    });
    items.sort(function(first, second) {
        return second[1][1] - first[1][1];
    });
    sorted_obj={}
    $.each(items, function(k, v) {
        use_key = v[0]
        use_value = v[1]
        sorted_obj[use_key] = use_value
    })
    return(sorted_obj)
} 


function getStartTime(region) {
    return region["start"]

}

/**
 * Align interval functions. Solves by a greedy strategy where we sort by start_times.
 * If the intervals are within a buffer (hardcoded to be 0.25) rewrite their start and end times so that they are cleanly concatenated.
 *
 */
function alignIntervals() {
  
    console.log("aligning intervals");
    let i = 0;
    let intervalDict = {}
    var lastInterval;
    
    var regions_start_times = Object.keys(wavesurfer.regions.list).map(function(key) {
        
        key_string = key.toString()
        intervalDict[ getStartTime(wavesurfer.regions.list[key])] = key_string
        i++;

        // return start_time_dict;
    });
  

    keys = Object.keys(intervalDict),
    i, len = keys.length;
    
    keys.sort();
    
    regions_by_start_times = Array();

    for (i = 0; i < len; i++) {
      k = keys[i];
      regions_by_start_times.push(intervalDict[k].toString());
    }

    j = 0;
    for (let sorted_key in regions_by_start_times) {
        // console.log(regions_by_start_times[sorted_key]);
        if (j != 0) {
            
            // let storeEnd = wavesurfer.regions.list[key][""];
            a = lastInterval;
            a_start = a["start"];
            a_end =  a["end"];
            b = wavesurfer.regions.list[regions_by_start_times[sorted_key]];
            b_start = b["start"];
            b_end = b["end"];
            
            // [A] [B]
            // PICK THE EARLIEST STRATEGY
            // if A ends BEFORE B starts and the buffer is smaller than 0.1 seconds... SNAP TO WHEN A ENDS
            if (a_start < b_start && a_end < b_start && Math.abs(a_end - b_start) < 0.25 ) {
                b["start"] = a_end;
                            
            }

            // if A ends AFTER B starts and the buffer is smaller than 0.1 seconds...
             if (a_start < b_start &&  b_start < a_end && Math.abs(a_end - b_start) < 0.25  ) {
                b["start"] = a_end;
            }
 
        }
        j++;

        lastInterval = wavesurfer.regions.list[regions_by_start_times[sorted_key]];
        console.log("last interval");
        console.log(sorted_key);
    }
    
    console.log("update intervals");
    j = 0;
    for (let sorted_key in regions_by_start_times) {
        console.log(wavesurfer.regions.list[regions_by_start_times[sorted_key]]);
               start =  console.log(wavesurfer.regions.list[regions_by_start_times[sorted_key]]["start"]);
                 end =        console.log(wavesurfer.regions.list[regions_by_start_times[sorted_key]]["end"]);
                    wavesurfer.regions.list[regions_by_start_times[sorted_key]].update({start:start,end:end});
    }
    //renumberIntervals();
    //saveRegions();

    //location.reload();

}

/**
 * Adds a new interval which is by default set to be 1 second. The start time is wherever the playhead is.
 *
 */
function addInterval() {
    wavesurfer.addRegion({start:wavesurfer.getCurrentTime(), end:wavesurfer.getCurrentTime()+1.0});

}

