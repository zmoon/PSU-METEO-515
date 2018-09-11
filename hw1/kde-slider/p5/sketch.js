
// preload table data
// function preload() {
//   var data = loadTable(
//     'PRSA-adapted-aparrish.csv',
//     'csv',
//     'header');
// }

let slider;
let prevSliderVal = 50;
let currSliderVal = 50;
let sliderMax = 500;

let n = 1000;
let xplotmin = 0;
let xplotmax = 420;
let xplot = new Array(n);
let yplot = new Array(n);

let c;

function setup() { 
  createCanvas(600, 400);
  background(220);
  stroke(50);
  textSize(24);

  slider = createSlider(0, sliderMax, currSliderVal);
  slider.position(40, 380);
  slider.style('width', '500px');

  for (let i = 0; i <= n; i++) {
    xplot[i] = i/n*(xplotmax-xplotmin) + xplotmin;
  }

  c = new Curve1(xplot, currSliderVal/sliderMax);
  console.log(c.x, c.y);

  noLoop();
} 

function draw() {  // the p5 draw function is what loops when loop() is set

  //   clear();
  background(220);

  if (mouseIsPressed || (keyIsPressed && keyCode === RIGHT_ARROW)) {  // in loop mode
    // loop();
    currSliderVal = slider.value();
    // console.log('currSliderVal:' + currSliderVal)

    c.calcY(currSliderVal/sliderMax);

  } else if (keyIsPressed){// && keyCode === RIGHT_ARROW) {
    console.log('blah '+keyCode)
  } else {  // out of loop mode
    // noLoop();
  }

  c.plotLine();

//   text('currSliderVal:', 10, 30);
//   text('prevSliderVal:', 10, 60);

//   xpos = 100 + currSliderVal / 2;
//   push();
//   fill('green');
//   text(currSliderVal, 170, 30);
//   ellipse(xpos, 205, 8, 8);
//   pop();

//   xpos = 100 + prevSliderVal / 2;
//   // console.log('xpos: '+xpos)
//   push();
//   fill('red');
//   text(prevSliderVal, 170, 60);
//   ellipse(xpos, 200, 8, 8);
//   pop();

//   console.log(currSliderVal)
//   prevSliderVal = currSliderVal;
}

function mousePressed() {
//   prevSliderVal = currSliderVal;
  loop();
}

function keyPressed() {
  if (keyCode === RIGHT_ARROW) {
    loop();
  }
}

function keyReleased() {
  if (keyCode === RIGHT_ARROW) {
    noLoop();
  }
}

function mouseReleased() {
  noLoop();
//   if (currSliderVal != prevSliderVal) {

    // c.calcY(currSliderVal/sliderMax);

    // console.log('  redrawing')
    // redraw(1);

    // prevSliderVal = currSliderVal;
  
//   }
}
