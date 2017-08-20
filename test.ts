//import {  Array2D, Graph,  NDArrayMathGPU, Scalar, Session, Tensor,SGDOptimizer,CostReduction,InGPUMemoryShuffledInputProviderBuilder } from '../deeplearnjs';
import {  Array2D, Graph,  NDArrayMathCPU, Scalar, Session, Tensor,SGDOptimizer,CostReduction,InCPUMemoryShuffledInputProviderBuilder } from '../deeplearnjs';

function leaky_relu(x: Tensor): Tensor {
  const mx: Tensor = graph.multiply(graph.constant(-1), graph.relu(x));
  const neg_part: Tensor = graph.multiply(graph.constant(-0.2), graph.relu(mx));
  return graph.add(graph.relu(x) , neg_part)
}


function mylog(txt: String) {
  console.log(txt);
  document.getElementById('output').innerHTML +=txt+"<BR>";
  
}

console.log('Start training..');

const graph = new Graph();

const x: Tensor = graph.placeholder('x', []);

const W1data = Array2D.randNormal([16, 1]);
const b1data = Array2D.zeros([16, 1]);
const W2data = Array2D.randNormal([32, 16],0,Math.sqrt(1.0/16.0));
const b2data = Array2D.zeros([32, 1]);
const W3data = Array2D.randNormal([1, 32],0,Math.sqrt(1.0/32.0));
const b3data = Array2D.zeros([1, 1]);

// const W1data = Array2D.randNormal([256, 1], 0, 1);
// const b1data = Array2D.zeros([256, 1]);
// const W2data = Array2D.randNormal([1024, 256],0, Math.sqrt(1.0/256.0));
// const b2data = Array2D.zeros([1024, 1]);
// const W3data = Array2D.randNormal([1, 1024], Math.sqrt(1.0/1024.0));
// const b3data = Array2D.zeros([1, 1]);

const W1: Tensor = graph.variable('W1', W1data);
const b1: Tensor = graph.variable('b1', b1data);
const W2: Tensor = graph.variable('W2', W2data);
const b2: Tensor = graph.variable('b2', b2data);
const W3: Tensor = graph.variable('W3', W3data);
const b3: Tensor = graph.variable('b3', b3data);

const h1: Tensor = leaky_relu(graph.add(graph.multiply(W1, x), b1));
const h2: Tensor = leaky_relu(graph.add(graph.matmul(W2, h1), b2));
const y_: Tensor = leaky_relu(graph.add(graph.matmul(W3, h2), b3));
const y: Tensor = graph.reshape(y_,[]);

const yLabel: Tensor = graph.placeholder('y label', []);
const cost: Tensor = graph.meanSquaredCost(y, yLabel);

//const math = new NDArrayMathGPU();
const math = new NDArrayMathCPU();

const session = new Session(graph, math);

math.scope((keep, track) => {
  
  var xs: Scalar[] = [];
  var ys: Scalar[] = [];
  
  for (var i = 0; i < 100; i++) {
    var xr = Math.random();
    xs.push(track(Scalar.new(xr)));
    ys.push(track(Scalar.new(Math.exp(xr))));
  }

  const shuffledInputProviderBuilder =
 //     new InGPUMemoryShuffledInputProviderBuilder([xs, ys]);
      new InCPUMemoryShuffledInputProviderBuilder([xs, ys]);
      const [xProvider, yProvider] =
      shuffledInputProviderBuilder.getInputProviders();

  const NUM_BATCHES = 100;
  const BATCH_SIZE = xs.length;
  const LEARNING_RATE = 0.01;
  const optimizer = new SGDOptimizer(LEARNING_RATE);

  var startTime = new Date();

  for (let i = 0; i < NUM_BATCHES; i++) {
      const costValue = session.train(
        cost,
        [{ tensor: x, data: xProvider }, { tensor: yLabel, data: yProvider }],
        BATCH_SIZE, optimizer, CostReduction.MEAN);
        var cost_val = costValue.get();
        console.log('average cost: ' + cost_val);
    }

  console.log('training finished.');

  var endTime = new Date();
  var timeDiff = endTime.getTime() - startTime.getTime();

  // for prediction check

  var predicted = session.eval(y, [{ tensor: x, data: track(Scalar.new(0.2)) }]).getValues();
  mylog('--- prediction check ---');
  mylog('predicted : ' + predicted);
  mylog('truth     :' + Math.exp(0.2));

  mylog('');
  mylog('--- benchmark result ---');
  mylog('elasped time for training: ' + timeDiff/1000 +'[sec]');

  // for prediction benchmark

  console.log('executing prediction benchmark..');
  
  var startTime = new Date();
  
  for (let i = 0; i < 1000; i++) {
    var xdata = Scalar.new(Math.random());
    var predicted = session.eval(y, [{ tensor: x, data: track(xdata) }]).getValues();
  }
  var endTime = new Date();
  var timeDiff = endTime.getTime() - startTime.getTime();

  mylog('elasped time for prediction: ' + timeDiff/1000.0 +'[msec/cycle]');
    
}

);

