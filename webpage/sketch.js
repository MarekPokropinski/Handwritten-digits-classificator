var buffer = new Array(28);

var painting = false;
let current;
var nn_model;
var isModelReady = false;
var timeToPredict = 10;

var plotDiv;
var plotData;
var plotLayout;

function reset() {
  for (let i = 0; i < 28; i++) {
    for (let j = 0; j < 28; j++) {
      buffer[i][j] = 0;
    }
  }
  background(255);
}

async function loadNNModel() {
  nn_model = await tf.loadLayersModel("/model.json");
  isModelReady = true;
  predict();
}

function setupCanvas() {
  const canvas = document.getElementById("defaultCanvas0");
  const canvasContainer = document.getElementById("canvasContainer");

  canvasContainer.appendChild(canvas);
}

function setup() {
  createCanvas(280, 280);
  setupCanvas();
  current = createVector(0, 0);
  for (let i = 0; i < 28; i++) {
    buffer[i] = new Array(28);
  }
  reset();
  frameRate(120);
  loadNNModel();
  plotDiv = document.getElementById("PLOT");
  plotData = [
    {
      x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      y: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      type: "bar"
    }
  ];
  plotLayout = {
    title: "prediction",
    xaxis: { dtick: 1, range: [-0.5, 9.5], title: "digit", fixedrange: true },
    yaxis: { range: [0, 1], title: "certainty", fixedrange: true }
  };
  Plotly.newPlot(plotDiv, plotData, plotLayout, { responsive: true });
}

function isInRange(x, y) {
  return x >= 0 && x < 28 && y >= 0 && y < 28;
}

function drawAt(x, y, val) {
  if (isInRange(x, y)) {
    buffer[y][x] = Math.min(255, buffer[y][x] + val);
    fill(255 - buffer[y][x]);
    square(x * 10, y * 10, 10);
  }
}

function plotPrediction(prediction) {
  plotData[0].y = prediction;
  Plotly.react(plotDiv, plotData, plotLayout, { responsive: true });
}

function predict() {
  if (!isModelReady) {
    return;
  }
  let d = [new Array(28)];

  for (let i = 0; i < 28; i++) {
    d[0][i] = new Array(28);
    for (let j = 0; j < 28; j++) {
      d[0][i][j] = [(buffer[i][j] - 33.318447) / (78.567444 + 1e-6)];
    }
  }
  tf.tidy(() => {
    const input = tf.tensor4d(d);
    const prediction = nn_model.predict(input);
    prediction.array().then(array => plotPrediction(array[0]));
  });
}

function draw() {
  noStroke();

  if (painting) {
    current.x = mouseX;
    current.y = mouseY;
    x = Math.round(current.x / 10 - 0.5);
    y = Math.round(current.y / 10 - 0.5);
    drawAt(x, y, 120);

    drawAt(x - 1, y, 120);
    drawAt(x, y - 1, 120);
    drawAt(x, y + 1, 120);
    drawAt(x + 1, y, 120);

    drawAt(x - 1, y - 1, 90);
    drawAt(x + 1, y - 1, 90);
    drawAt(x - 1, y + 1, 90);
    drawAt(x + 1, y + 1, 90);
    if (timeToPredict == 0) {
      predict();
      timeToPredict = 10;
    }
    timeToPredict -= 1;
  }
}

function mousePressed() {
  painting = true;
}

// Stop
function mouseReleased() {
  painting = false;

  predict();
}

touchStarted = mousePressed;
touchEnded = mouseReleased;
