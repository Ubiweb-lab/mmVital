/**
 * return an option for chart.
 * @param {*} type , 0: HR;  1: RR
 */
var getOption = function (type, x_data) {

    let option = {
        title: {
            text: type == 0 ? 'BPM' : 'RPM',
            left: 'right',
            textStyle: {
              color: 'rgb(255 255 255)',
              fontSize: 14,
              fontWeight:'regular',
            },
            //subtext: 'BPM'
        },
        grid: {
            left: '8%',   
            right: '5%',  
            top: '10%',    
            bottom: '15%'  
        },
        xAxis: {
            type: 'category',
            boundaryGap: false,
            data: x_data,
            show: true,
            name: 'S',
            axisLine: {
                lineStyle: {
                    color: 'rgba(38, 38, 38, 1)', // Set the color of the x-axis line
                    fontSize: 10,
                }
            },
            axisLabel: {
                interval: 9, // Set the interval to control the length of the xAxis
                color: 'rgb(176 176 177)',  // Set the font color of the X-axis tick labels
                formatter: function (value, index) {
                    if (index === 60) {
                      return value + ' s';
                    } else {
                      return value;
                    }
                },
            },
        },
        yAxis: {
            type: 'value',
            name: 'BPM',
            min: type==0? 40 : 5,
            max: type==0? 100 : 40,
            splitLine: {
                show: true,
                lineStyle: {
                    color: 'rgba(38, 38, 38, 1)'  // Set the color of the split line
                }
            },
            axisLabel: {
                fontSize: 10, // Set the font size of the Y-axis labels
                color: 'rgb(176 176 177)',
            }
        },
        series: [
            {
                name: type == 0 ? 'BPM' : 'RPM',
                data: [],
                type: 'line',
                smooth: true,
                showSymbol: false,
                lineStyle: {
                    width: 1,
                    color: 'rgb(197 102 82)',
                    emphasis: {
                      color: 'rgb(197 102 82)',
                    }
                  }
            }
        ]
    };
    return option;
}