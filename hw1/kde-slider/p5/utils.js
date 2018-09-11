
// could create class for curve fn with the math part with params 
// and string form printing the equation

// maybe these should be set in the sketch
let xplotoffset = 80;
let mapya = 350;
let mapyb = 70;

class Curve1 {
  constructor(x, a) {
    this.x = x;  // array of x values
    this.n = x.length; 
    this.y = new Array(n);  // actual calculated y values
    this.ym = new Array(n);  // y values mapped to desired display area

    this.calcY(a);
    
  }

  calcY(a) {
    for (let i = 0; i < this.n; i++) {
      this.y[i] = this.x[i]**(1+a*2);  // the fn could be part of constructor of a parent class
    }
    this.ymin = min(this.y);
    this.ymax = max(this.y);
    this.a = a;

    return this.y;
  }

  mapY(ya, yb) {

    // this.ym = map(this.y, this.ymin, this.ymax, ya, yb);

    for (let i = 0; i < this.n; i++) {
      this.ym[i] = map(this.y[i], this.ymin, this.ymax, ya, yb);
    }

    return this.ym;
  }

  plotLine(eqn=true) {
    push();
    stroke(100);
    textSize(18);

    yplot = this.mapY(mapya, mapyb);
    for (let i = 0; i < this.n-1; i++) {
      line(this.x[i]+xplotoffset, yplot[i], 
        this.x[i+1]+xplotoffset, yplot[i+1])
    }

    if (eqn) {
      text('y = x^'+(1+this.a*2).toFixed(3), 450, 300)
    }
    
    pop();
  }

}




