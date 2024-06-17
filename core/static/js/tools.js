const START_DATE = new Date();
const START_HOURS = START_DATE.getHours() < 10 ? "0" + START_DATE.getHours() : START_DATE.getHours();

const PERIOD = START_HOURS < 12 ? 'AM' : 'PM';

const START_MINUTES = START_DATE.getMinutes() < 10 ? "0" + START_DATE.getMinutes() : START_DATE.getMinutes();
const START_SECONDS = START_DATE.getSeconds();
const START_TIME = START_HOURS + ":" + START_MINUTES;

let connectionStatus = 1;

/**
 * set the start time of mornitoring.
 */
var setStartTime = function () {
    $("#startTime").text(START_TIME);
    $("#period").text(PERIOD);
}

/**
 * refresh the duration time every 1 second.
 */
var updateDurationTime = function () {

    function refreshTime() {
        if(connectionStatus == 0) {
            clearInterval(timeInterval);
        }
        let currentDate = new Date();
        let timeDifference = currentDate - START_DATE;

        let seconds = Math.floor(timeDifference / 1000);
        seconds = seconds % 60;
        seconds = seconds < 10 ? "0" + seconds : seconds;

        let minutes = Math.floor(timeDifference / (1000 * 60));
        minutes = minutes % 60;
        minutes = minutes < 10 ? "0" + minutes : minutes;

        let hours = Math.floor(timeDifference / (1000 * 60 * 60));
        hours = hours < 10 ? "0" + hours : hours;

        let durationTime = hours + ":" + minutes + ":" + seconds;

        $("#durationTime").text(durationTime);

    }
    let timeInterval = setInterval(refreshTime, 1000);
}

/**
 * refresh HR, RR, min, max, and avg. 
 */
var refreshHrAndRr = function () {

    let hr_array = [];
    let rr_array = [];
    
    // initialize the data of x-axis.
    const X_LENGTH = 60;
    let x_data = [];
    for(var i=0; i<=X_LENGTH; i++) {
        x_data.push(i);
    }

    let min_hr = 0, min_rr = 0, avg_hr = 0, avg_rr = 0, max_hr = 0, max_rr = 0;
    let current_hr = 0, current_rr = 0;
    let index_id = 0;
    let isUpdateYscale = false;

    function refresh() {
        var csrftoken = $("[name=csrfmiddlewaretoken]").val();
        $.ajaxSetup({
            headers: { "X-CSRFToken": csrftoken }
        });
        $.ajax({
            url: "../refresh_hr_and_rr/",
            type: "POST",
            dataType: "JSON",
            data: {
                index_id: index_id,
            },
            success: function (data) {
                index_id++;

                current_hr = data.HR;
                current_rr = data.RR;

                hr_array.push(current_hr);
                rr_array.push(current_rr);

                if (min_hr == 0 | min_hr > current_hr) { // update min_hr
                    min_hr = current_hr;
                }
                if (max_hr == 0 | max_hr < current_hr) {
                    max_hr = current_hr;
                }
                if (min_rr == 0 | min_rr > current_rr) { // update min_rr
                    min_rr = current_rr;
                }
                if (max_rr == 0 | max_rr < current_rr) {
                    max_rr = current_rr;
                }
                const sum_hr = hr_array.reduce((total, nextValue) => total + nextValue, 0);
                const sum_rr = rr_array.reduce((total, nextValue) => total + nextValue, 0);

                avg_hr = Math.round(sum_hr / hr_array.length);
                avg_rr = Math.round(sum_rr / rr_array.length);

                $("#realTime_HR").text(current_hr);
                $("#realTime_RR").text(current_rr);

                $("#min_hr").text(min_hr);
                $("#max_hr").text(max_hr);
                $("#min_rr").text(min_rr);
                $("#max_rr").text(max_rr);

                $("#avg_hr").text(avg_hr);
                $("#avg_rr").text(avg_rr);

                $("#statusTextHr").html("Signal Receiving...");
                $("#statusTextRr").html("Signal Receiving...");

                // dynamically set the x-scale.
                if(index_id >= x_data[x_data.length-1]) {
                    let splice_length = 1;
                    x_data.splice(0, splice_length);
                    hr_array.splice(0, splice_length);
                    rr_array.splice(0, splice_length);
                    for(var i=index_id; i<(index_id+splice_length); i++) {
                        x_data.push(i+1);
                    }
                    // isUpdateYscale = true;
                } 
                renderWaves(document.getElementById('hr_wave'), 0, x_data, hr_array, isUpdateYscale);
                renderWaves(document.getElementById('rr_wave'), 1, x_data, rr_array, isUpdateYscale);
            },
            error: function (xhr, status, error) {
                connectionStatus = 0;
                $("#statusTextHr").html("<span class='text-red-500'>Connection Ended</span>");
                $("#statusTextRr").html("<span class='text-red-500'>Connection Ended</span>");
                
                clearInterval(refreshInterval);
            }
        });
    }
    let refreshInterval = setInterval(refresh, 1000);
}

/**
 * render HR and RR waves.
 * @param {*} container 
 */
var renderWaves = function (container, type, x_data, y_data, isUpdateYscale) {
    let waveChart = echarts.init(container);

    let option = getOption(type);
    option.series[0].data = y_data;
    option.xAxis.data = x_data;

    // dynamically adape the y-scale
    if (isUpdateYscale) {
        let max = Math.max(...y_data);
        max = max + (10 - max % 10) //+ 10;
        option.yAxis.max = max;

        let min = Math.min(...y_data);
        min = min - min % 10 //- 10;
        option.yAxis.min = min;
    }
    
    waveChart.setOption(option);
}