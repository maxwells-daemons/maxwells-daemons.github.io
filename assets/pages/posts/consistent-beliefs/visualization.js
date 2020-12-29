const TOP_MARGIN = 40;
const BOTTOM_MARGIN = 60;
const FRAME_SIZE = 500;
const FRAME_WIDTH = 1;
const POINT_SIZE = 5;

const [NUM_UL, NUM_UR, NUM_LL, NUM_LR] = [50, 10, 10, 10];

// From:
// https://stackoverflow.com/questions/25582882/javascript-math-random-normal-distribution-gaussian-bell-curve
function randn_bm() {
  var u = 0, v = 0;
  while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
  while(v === 0) v = Math.random();
  return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function resampled_normal(mean, variance, min, max) {
  while(true) {
    let value = randn_bm() * variance + mean;
    if ((value >= min) && (value <= max)) {
      return value;
    }
  }
}

const svg = d3.select('#visualization').append('svg')
  .attr('width', FRAME_SIZE + 2 * FRAME_WIDTH)
  .attr('height', FRAME_SIZE + 2 * FRAME_WIDTH + TOP_MARGIN + BOTTOM_MARGIN);

const background = svg.append('rect')
  .attr('x', FRAME_WIDTH).attr('y', FRAME_WIDTH + TOP_MARGIN)
  .attr('width', FRAME_SIZE).attr('height', FRAME_SIZE)
  .style('fill', 'white');

const scaleX = d3.scaleLinear().domain([-1, 1]).range([FRAME_WIDTH, FRAME_WIDTH + FRAME_SIZE]);
const scaleY = d3.scaleLinear()
  .domain([-1, 1]).range([FRAME_WIDTH + FRAME_SIZE + TOP_MARGIN, FRAME_WIDTH + TOP_MARGIN]);

const xAxis = svg.append('line')
  .attr('x1', scaleX(-1)).attr('x2', scaleX(1))
  .attr('y1', scaleY(0)).attr('y2', scaleY(0))
  .style('stroke', 'grey').style('stroke-width', 1);

const yAxis = svg.append('line')
  .attr('x1', scaleX(0)).attr('x2', scaleX(0))
  .attr('y1', scaleY(-1)).attr('y2', scaleY(1))
  .style('stroke', 'grey').style('stroke-width', 1);

function setCenter(selection) {
  return selection
    .attr('cx', (d, _) => scaleX(d[0]))
    .attr('cy', (d, _) => scaleY(d[1]));
}

let [pointsUL, pointsUR, pointsLL, pointsLR] = [[], [], [], []];
function updatePoints() {
  pointsUL = [...Array(NUM_UL)].map(() => [resampled_normal(-0.5, 0.15, -0.95, -0.05),
                                           resampled_normal(0.5, 0.15, 0.05, 0.95)]);
  svg.selectAll('.pointUL')
    .data(pointsUL)
    .join(
      enter => enter.append('circle').attr('r', POINT_SIZE)
        .attr('class', 'pointUL')
        .style('fill', 'red').style('fill-opacity', 0.3)
        .style('stroke', 'red').style('stroke-width', 1)
        .call(setCenter),
      update => update.call(u => setCenter(u.transition())));

  pointsUR = [...Array(NUM_UR)].map(() => [resampled_normal(0.65, 0.15, 0.05, 0.95),
                                           resampled_normal(0.35, 0.15, 0.05, 0.95)]);
  svg.selectAll('.pointUR')
    .data(pointsUR)
    .join(
      enter => enter.append('circle').attr('r', POINT_SIZE)
        .attr('class', 'pointUR')
        .style('fill', 'blue').style('fill-opacity', 0.3)
        .style('stroke', 'blue').style('stroke-width', 1)
        .call(setCenter),
      update => update.call(u => setCenter(u.transition())));

  pointsLL = [...Array(NUM_LL)].map(() => [resampled_normal(-0.35, 0.15, -0.95, -0.05),
                                           resampled_normal(-0.65, 0.15, -0.95, -0.05)]);
  svg.selectAll('.pointLL')
    .data(pointsLL)
    .join(
      enter => enter.append('circle').attr('r', POINT_SIZE)
        .attr('class', 'pointLL')
        .style('fill', 'green').style('fill-opacity', 0.3)
        .style('stroke', 'green').style('stroke-width', 1)
        .call(setCenter),
      update => update.call(u => setCenter(u.transition())));

  pointsLR = [...Array(NUM_LR)].map(() => [resampled_normal(0.5, 0.15, 0, 1),
                                           resampled_normal(-0.5, 0.15, -1, 0)]);
  svg.selectAll('.pointLR')
    .data(pointsLR)
    .join(
      enter => enter.append('circle').attr('r', POINT_SIZE)
        .attr('class', 'pointLR')
        .style('fill', 'orange').style('fill-opacity', 0.3)
        .style('stroke', 'orange').style('stroke-width', 1)
        .call(setCenter),
      update => update.call(u => setCenter(u.transition())));
}
updatePoints();

const mouseLine = d3.lineRadial();
const mouseLinePath = svg.append('path')
  .attr('transform', 'translate(' + scaleX(0) + ',' + scaleY(0) + ')')
  .attr('stroke', 'black ').attr('stroke-width', 2);
const mouseLineLength = Math.sqrt(2) * FRAME_SIZE;

let [mouseX, mouseY] = [undefined, undefined];
let mouseAngle = undefined;
function drawLine(mouseX, mouseY) {
  mouseX = scaleX.invert(mouseX);
  mouseY = scaleY.invert(mouseY);
  mouseAngle = Math.atan2(mouseX, mouseY);
  mouseLinePath.attr('d', mouseLine([[mouseAngle, mouseLineLength], [mouseAngle, -mouseLineLength]]));
}

function aboveLine(point) {
  const [x, y] = point;
  const slope = Math.tan(-1 * mouseAngle + Math.PI / 2);
  return y > x * slope;
}


const topFiller = svg.append('rect')
  .attr('x', 0).attr('y', 0)
  .attr('width', FRAME_SIZE + 2 * FRAME_WIDTH)
  .attr('height', TOP_MARGIN + 1)
  .style('fill', '#f5f5f5');

const topBar = svg.append('rect')
  .attr('x', 0).attr('y', 0)
  .attr('width', FRAME_SIZE + 2 * FRAME_WIDTH)
  .attr('height', TOP_MARGIN - 5)
  .style('fill', 'white');


const bottomFiller = svg.append('rect')
  .attr('x', 0).attr('y', FRAME_SIZE + FRAME_WIDTH + TOP_MARGIN)
  .attr('width', FRAME_SIZE + 2 * FRAME_WIDTH)
  .attr('height', BOTTOM_MARGIN + 1)
  .style('fill', '#f5f5f5');

const resampleButton = svg.append("rect")
  .attr('x', FRAME_SIZE + FRAME_WIDTH - 85).attr('y', FRAME_SIZE + TOP_MARGIN + 5)
  .attr('width', 85).attr('height', 35)
  .style('fill', 'white')
  .style('stroke', 'black')
  .style('stroke-width', 1)
  .attr('rx', 2)
  .attr('ry', 2);
const resampleText = svg.append("text")
  .attr("x", FRAME_SIZE + FRAME_WIDTH - 80).attr("y", FRAME_SIZE + TOP_MARGIN + BOTTOM_MARGIN / 2 - 1)
  .text("Resample")
  .attr("pointer-events", "none");
resampleButton.on('click', updatePoints);

const lossScale = d3.scaleLinear().domain([0, 0.5]).range([0, FRAME_SIZE]);
const lossBar = svg.append('rect')
  .attr('x', 0).attr('y', 0)
  .attr('height', TOP_MARGIN - 5)
  .style('fill', '#90caf9');
const lossText = svg.append('text')
  .attr('x', 10).attr('y', TOP_MARGIN / 2 + 5)
  .attr('font-size', 20)
  .attr("pointer-events", "none");

const diffPairs = NUM_UL * (NUM_UR + NUM_LL + NUM_LR) + NUM_UR * (NUM_LL + NUM_LR) + NUM_LL * NUM_LR;
function computeLoss() {
  let countUL = pointsUL.map(aboveLine).reduce((acc, val) => acc + val);
  let countUR = pointsUR.map(aboveLine).reduce((acc, val) => acc + val);
  let countLR = pointsLR.map(aboveLine).reduce((acc, val) => acc + val);
  let countLL = pointsLL.map(aboveLine).reduce((acc, val) => acc + val);
  const diffAbove = countUL * (countUR + countLR + countLL) +
                   countUR * (countLR + countLL) +
                   countLR * countLL;
  countUL = NUM_UL - countUL;
  countUR = NUM_UR - countUR;
  countLR = NUM_LR - countLR;
  countLL = NUM_LL - countLL;
  const diffBelow = countUL * (countUR + countLR + countLL) +
                   countUR * (countLR + countLL) +
                   countLR * countLL;
  const loss = (diffAbove + diffBelow) / diffPairs;

  lossBar.attr('width', lossScale(loss));
  lossText.text('Loss: ' + Math.round(100 * loss) / 100);
}

const frame = svg.append('rect')
  .attr('x', FRAME_WIDTH).attr('y', FRAME_WIDTH + TOP_MARGIN)
  .attr('width', FRAME_SIZE).attr('height', FRAME_SIZE)
  .style('fill', 'none').style('stroke', 'black').style('stroke-width', FRAME_WIDTH);

function drawMouse(mouseX, mouseY) {
  drawLine(mouseX, mouseY);
  computeLoss();
}
function mouseHandler() {
  [mouseX, mouseY] = d3.mouse(this);
  drawMouse(mouseX, mouseY);
}
drawMouse(scaleX.invert(1), scaleY.invert(1));
background.on('mousemove', mouseHandler);
