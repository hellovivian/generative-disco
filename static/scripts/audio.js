'use strict';

// Create an instance
var wavesurfer;

// function waveSurferInit(audioSrc) {
//     wavesurfer = WaveSurfer.create({
//         container: document.querySelector('#waveform'),
//         height: 100,
//         pixelRatio: 1,
//         minPxPerSec: 100,
//         scrollParent: true,
//         normalize: true,
//         splitChannels: false,
//         barWidth: 3,
//         barRadius: 3,
//         cursorWidth: 1,
//         height: 200,
//         barGap: 3,
//         // backend: 'MediaElement',
//         plugins: [
//             // WaveSurfer.regions.create(),
//             WaveSurfer.regions.create({
   
//                 regions: [
//                     {
//                         start: 0,
//                         end: 0.5,
//                         loop: false,
//                         color: 'hsla(400, 100%, 30%, 0.5)'
//                     }, {
//                         start: 0.75,
//                         end: 1,
//                         loop: false,
//                         color: 'hsla(200, 50%, 70%, 0.4)'
//                     }
//                 ],
//                 dragSelection: {
//                     slop: 5
//                 }
//             }),
//             WaveSurfer.timeline.create({
//                 container: '#wave-timeline'
//             }),
//             WaveSurfer.cursor.create()
//         ]});

//         wavesurfer.load(audioSrc);

// }


// Init & load audio file
// document.addEventListener('DOMContentLoaded', function() {
//     let options = {
//         container: document.querySelector('#waveform'),
//         waveColor: 'violet',
//         progressColor: 'purple',
//         cursorColor: 'navy'
//     };


//     // Init
//     wavesurfer = WaveSurfer.create(options);
//     // Load audio from URL
//     wavesurfer.load('../static/audio/gaga_trimmed.wav');

//     // Regions
//     if (wavesurfer.enableDragSelection) {
//         wavesurfer.enableDragSelection({
//             color: 'rgba(0, 255, 0, 0.1)'
//         });
//     }
// });

// Play at once when ready
// Won't work on iOS until you touch the page
wavesurfer.on('ready', function() {
    //wavesurfer.play();
});

// Report errors
wavesurfer.on('error', function(err) {
    console.error(err);
});

// Do something when the clip is over
wavesurfer.on('finish', function() {
    console.log('Finished playing');
});

/* Progress bar */
document.addEventListener('DOMContentLoaded', function() {
    const progressDiv = document.querySelector('#progress-bar');
    const progressBar = progressDiv.querySelector('.progress-bar');

    let showProgress = function(percent) {
        progressDiv.style.display = 'block';
        progressBar.style.width = percent + '%';
    };

    let hideProgress = function() {
        progressDiv.style.display = 'none';
    };

    wavesurfer.on('loading', showProgress);
    wavesurfer.on('ready', hideProgress);
    wavesurfer.on('destroy', hideProgress);
    wavesurfer.on('error', hideProgress);
});

// Drag'n'drop
document.addEventListener('DOMContentLoaded', function() {
    let toggleActive = function(e, toggle) {
        e.stopPropagation();
        e.preventDefault();
        toggle
            ? e.target.classList.add('wavesurfer-dragover')
            : e.target.classList.remove('wavesurfer-dragover');
    };

    let handlers = {
        // Drop event
        drop: function(e) {
            toggleActive(e, false);

            // Load the file into wavesurfer
            if (e.dataTransfer.files.length) {
                wavesurfer.loadBlob(e.dataTransfer.files[0]);
            } else {
                wavesurfer.fireEvent('error', 'Not a file');
            }
        },

        // Drag-over event
        dragover: function(e) {
            toggleActive(e, true);
        },

        // Drag-leave event
        dragleave: function(e) {
            toggleActive(e, false);
        }
    };

    let dropTarget = document.querySelector('#drop');
    Object.keys(handlers).forEach(function(event) {
        dropTarget.addEventListener(event, handlers[event]);
    });
});

let ws = window.wavesurfer;

var GLOBAL_ACTIONS = { // eslint-disable-line
    play: function() {
        window.wavesurfer.playPause();
    },

    back: function() {
        window.wavesurfer.skipBackward();
    },

    forth: function() {
        window.wavesurfer.skipForward();
    },

    'toggle-mute': function() {
        window.wavesurfer.toggleMute();
    }
};

// Bind actions to buttons and keypresses
document.addEventListener('DOMContentLoaded', function() {
    document.addEventListener('keydown', function(e) {
        let map = {
            32: 'play', // space
            37: 'back', // left
            39: 'forth' // right
        };
        let action = map[e.keyCode];
        if (action in GLOBAL_ACTIONS) {
            if (document == e.target || document.body == e.target || e.target.attributes["data-action"]) {
                e.preventDefault();
            }
            GLOBAL_ACTIONS[action](e);
        }
    });

    [].forEach.call(document.querySelectorAll('[data-action]'), function(el) {
        el.addEventListener('click', function(e) {
            let action = e.currentTarget.dataset.action;
            if (action in GLOBAL_ACTIONS) {
                e.preventDefault();
                GLOBAL_ACTIONS[action](e);
            }
        });
    });
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







// Init & load audio file
document.addEventListener('DOMContentLoaded', function() {
    // Init
    wavesurfer = WaveSurfer.create({
        container: document.querySelector('#waveform'),
        height: 100,
        pixelRatio: 1,
        minPxPerSec: 100,
        scrollParent: true,
        normalize: true,
        splitChannels: false,
        backend: 'MediaElement',
        plugins: [
            WaveSurfer.regions.create(),
            WaveSurfer.minimap.create({
                height: 30,
                waveColor: '#ddd',
                progressColor: '#999'
            }),
            WaveSurfer.timeline.create({
                container: '#wave-timeline'
            }),
            WaveSurfer.cursor.create()
        ]
    });

    wavesurfer.on('ready', function() {
        wavesurfer.enableDragSelection({
            color: randomColor(0.25)
        });

        wavesurfer.util
            .fetchFile({
                responseType: 'json',
                url: './static/example.json'
            })
            .on('success', function(data) {
                alert("success");
                loadRegions(data);
                saveRegions();
            });
    });


    wavesurfer.on('region-click', function(region, e) {
        e.stopPropagation();
        // Play on click, loop on shift click
        // e.shiftKey ? region.playLoop() : region.play();
    });
    wavesurfer.on('region-click', editAnnotation);
    wavesurfer.on('region-update-end', saveRegions);
    wavesurfer.on('region-updated', saveRegions);
    wavesurfer.on('region-removed', saveRegions);
    wavesurfer.on('region-in', showNote);
    wavesurfer.on('region-out', hideNote);

    wavesurfer.on('region-play', function(region) {
        region.once('out', function() {
            wavesurfer.play(region.start);
            wavesurfer.pause();
        });
    });

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
 * Save annotations to localStorage.
 */
function saveRegions() {
    localStorage.regions = JSON.stringify(
        Object.keys(wavesurfer.regions.list).map(function(id) {
            let region = wavesurfer.regions.list[id];
            return {
                start: region.start,
                end: region.end,
                attributes: region.attributes,
                data: region.data
            };
        })
    );
}

/**
 * Load regions from localStorage.
 */
function loadRegions(regions) {
    regions.forEach(function(region) {
        region.color = randomColor(0.25);
        wavesurfer.addRegion(region);
    });
}

/**
 * Extract regions separated by silence.
 */
function extractRegions(peaks, duration) {
    // Silence params
    let minValue = 0.0015;
    let minSeconds = 0.25;

    let length = peaks.length;
    let coef = duration / length;
    let minLen = minSeconds / coef;

    // Gather silence indeces
    let silences = [];
    Array.prototype.forEach.call(peaks, function(val, index) {
        if (Math.abs(val) <= minValue) {
            silences.push(index);
        }
    });

    // Cluster silence values
    let clusters = [];
    silences.forEach(function(val, index) {
        if (clusters.length && val == silences[index - 1] + 1) {
            clusters[clusters.length - 1].push(val);
        } else {
            clusters.push([val]);
        }
    });

    // Filter silence clusters by minimum length
    let fClusters = clusters.filter(function(cluster) {
        return cluster.length >= minLen;
    });

    // Create regions on the edges of silences
    let regions = fClusters.map(function(cluster, index) {
        let next = fClusters[index + 1];
        return {
            start: cluster[cluster.length - 1],
            end: next ? next[0] : length - 1
        };
    });

    // Add an initial region if the audio doesn't start with silence
    let firstCluster = fClusters[0];
    if (firstCluster && firstCluster[0] != 0) {
        regions.unshift({
            start: 0,
            end: firstCluster[firstCluster.length - 1]
        });
    }

    // Filter regions by minimum length
    let fRegions = regions.filter(function(reg) {
        return reg.end - reg.start >= minLen;
    });

    // Return time-based regions
    return fRegions.map(function(reg) {
        return {
            start: Math.round(reg.start * coef * 100) / 100,
            end: Math.round(reg.end * coef * 100) / 100
        };
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
 * Edit annotation for a region.
 */
function editAnnotation(region) {
    let form = document.forms.edit;
    form.style.opacity = 1;
    (form.elements.start.value = Math.round(region.start * 100) / 100),
    (form.elements.end.value = Math.round(region.end * 100) / 100);
    form.elements.note.value = region.data.note || '';
    form.onsubmit = function(e) {
        e.preventDefault();
        region.update({
            start: form.elements.start.value,
            end: form.elements.end.value,
            data: {
                note: form.elements.note.value
            }
        });
        form.style.opacity = 0;
    };
    form.onreset = function() {
        form.style.opacity = 0;
        form.dataset.region = null;
    };
    form.dataset.region = region.id;
}

/**
 * Display annotation.
 */
function showNote(region) {
    if (!showNote.el) {
        showNote.el = document.querySelector('#subtitle');
    }
    showNote.el.style.color = 'Red';
    showNote.el.style.fontSize = 'large';
    showNote.el.textContent = region.data.note || '–';
}

function hideNote(region) {
    if (!hideNote.el) {
        hideNote.el = document.querySelector('#subtitle');
    }
    hideNote.el.style.color = 'Red';
    hideNote.el.style.fontSize = 'large';
    hideNote.el.textContent = '–';
}

/**
 * Bind controls.
 */
window.GLOBAL_ACTIONS['delete-region'] = function() {
    let form = document.forms.edit;
    let regionId = form.dataset.region;
    if (regionId) {
        wavesurfer.regions.list[regionId].remove();
        form.reset();
    }
};

window.GLOBAL_ACTIONS['export'] = function() {
    window.open(
        'data:application/json;charset=utf-8,' +
            encodeURIComponent(localStorage.regions)
    );
};

