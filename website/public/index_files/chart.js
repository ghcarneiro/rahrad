$(function () {
    $('#containerer3').highcharts({
        title: {
            text: 'Stability of 25-man guilds per tier',
            x: -20 //center
        },
        xAxis: {
            categories: ['Tier 14', 'Tier 15', 'Tier 16']
        },
        yAxis: {
	    max: 5000,
	    tickInterval: 1000,
            title: {
                text: 'Number of guilds'
            },
            plotLines: [{
                value: 0,
                width: 1,
                color: '#808080'
            }]
        },
        tooltip: {
            valueSuffix: ''
        },
        legend: {
            layout: 'vertical',
            align: 'right',
            verticalAlign: 'middle',
            borderWidth: 0
        },
        series: [{
            name: 'Total 25-man guilds',
            data: [1450, 1466, 4685]
        }, {
            name: 'Stable 25-man guilds',
            data: [1186, 1169, 2718]
        }]
    });
});

$(function () {
    $('#containerer').highcharts({
        chart: {
            type: 'column'
        },
        title: {
            text: 'Lifespan of 25-man guilds since Tier 14'
        },
        xAxis: {
            categories: [
                '1',
                '2',
                '3',
                '4',
                '5'
            ],
            crosshair: true
        },
        yAxis: {
            min: 0,
            title: {
                text: 'Number of guilds'
            }
        },
        tooltip: {
            headerFormat: '<span style="font-size:10px">Raided for {point.key} tiers</span><table>',
            pointFormat: '<tr><td style="color:{series.color};padding:0">Guilds: </td>' +
                '<td style="padding:0"><b>{point.y:.0f}</b></td></tr>',
            footerFormat: '</table>',
            shared: true,
            useHTML: true
        },
        plotOptions: {
            column: {
                pointPadding: 0.2,
                borderWidth: 0
            }
        },
        series: [{
            name: 'Number of tiers raided',
            data: [2385, 1667, 959, 280, 330]

        }]
    });
});
