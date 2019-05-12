import { Component, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { model } from '@tensorflow/tfjs';

@Component
({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})

export class AppComponent implements OnInit 
{
  title = 'TFJS-MobileNet';
  
  private picture: HTMLImageElement;
  private video: HTMLVideoElement;

  private csvContent: any;
  public trainData:  any[]=[];
  public parsedCsv: any[][];
  public csv: HTMLInputElement;

  private label: number[]=[];
  private xmin: number[]=[];
  private xmax: number[]=[];
  private ymin: number[]=[];
  private ymax: number[]=[];

  public src: string[]=[];

  
  ngOnInit()
  { 
    this.webcam_init();
    this.trainData=[];
    this.picture = <HTMLImageElement> document.getElementById("img");
    this.csv = <HTMLInputElement> document.getElementById("text") ;
  }

  customLossFunction(yTrue, yPred) 
  {
    const LABEL_MULTIPLIER = [224, 1, 1, 1, 1];
    return tf.tidy(() => 
    {
      //const cls_yTrue=yTrue.slice([0],[1]);
      //const cls_yPred=yPred.slice([0],[1]);
      //const rgr_yTrue=yTrue.slice([1],[4]);
      //const rgr_yPred=yPred.slice([1],[4]);      

      //console.log('binary Cross Entropy');
      //tf.metrics.binaryCrossentropy(cls_yTrue,cls_yPred).print();
      //console.log('meansquared');
      //tf.metrics.meanSquaredError(rgr_yTrue,rgr_yPred).print();
      
      //let loss = tf.add(tf.metrics.binaryCrossentropy(cls_yTrue,cls_yPred), tf.metrics.meanSquaredError(rgr_yTrue,rgr_yPred));

      const loss=tf.metrics.meanSquaredError(yTrue, yPred);
      console.log('binary cross entropy');
      console.log(loss);
      
      return loss;
    });
  }
 

  webcam_init()
  {  
    this.video = <HTMLVideoElement> document.getElementById("vid");
    
    navigator.mediaDevices.getUserMedia
    ({
      audio: false,
      video: 
      {
        facingMode: "user",
      }
    })
    
    .then(stream => 
      {
        this.video.srcObject = stream;
        this.video.onloadedmetadata = () => 
        {
          this.video.play();
        };
      });
  }

  detectFrame = (video, model) => {
    let vid=tf.browser.fromPixels(video);
    vid = vid.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    vid = vid.reshape([ 224, 224, 3]).expandDims(0);

    vid.print();
    const predictions = model.predict(vid).dataSync();
    this.renderPrediction(predictions);

    requestAnimationFrame(() => {
        this.detectFrame(this.video, model);
    });

  }

  public async predict() 
  {
    const loadedModel = <tf.LayersModel> await tf.loadLayersModel('../assets/my-model-1.json'); 
    loadedModel.summary();
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    mobilenet.summary();

    //loadedModel.summary();
    await this.detectFrame(this.video,loadedModel);
  }

  renderPrediction = (result) => {
    
    const canvas = <HTMLCanvasElement> document.getElementById("canvas");
    const showImage = <HTMLImageElement> document.getElementById('1');

    console.log(result);
    const ctx = canvas.getContext("2d");
    
    let predictionClass;

    canvas.width  = 224;
    canvas.height = 224;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
    //ctx.drawImage(showImage,0, 0,224,224);
    ctx.drawImage(this.video,0, 0,224,224);
    if (Number(result[0]>6))
    {
      predictionClass="not mouse"   
    }

    if (Number(result[0])<6)
    {
      predictionClass="mouse"   
    }

    const xmin = Number(result[1]);
    const ymin = Number(result[2]);
    const xmax = Number(result[3]);
    const ymax = Number(result[4]);

    const x = xmin; //top left corner of rectange
    const y = ymin  //top left corner of rectange
    
    const width = xmax-xmin;
    const height= ymax-ymin;

    console.log(x);
    console.log(y);
    
    console.log(width);
    console.log(height);
    
    ctx.strokeStyle = "#00FFFF";
    ctx.lineWidth = 4;
    ctx.strokeRect(x, y, width, height);
    // Draw the label background.
    ctx.fillStyle = "#00FFFF";
    const textWidth = ctx.measureText(predictionClass).width;
    const textHeight = parseInt(font, 10); // base 10
    ctx.fillRect(x, y, textWidth + 4, textHeight + 4);

    ctx.fillStyle = "#000000";
    ctx.fillText(predictionClass, x, y);

    console.log(result);
  }
  
  async loadTruncatedBase() 
  {
    const topLayerGroupNames = [ 'conv_pw_11'];
    const topLayerName =`${topLayerGroupNames[topLayerGroupNames.length - 1]}_relu`;

    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'); 
    
    const fineTuningLayers = [];
    const layer = mobilenet.getLayer(topLayerName);
    const truncatedBase = tf.model({inputs: mobilenet.inputs, outputs: layer.output, name: 'modelTruncated'});

    console.log(truncatedBase.summary());

    for (const layer of truncatedBase.layers) 
    {

      layer.trainable = false;

      for (const groupName of topLayerGroupNames) 
      {
        if (layer.name.indexOf(groupName) === 0) 
        {
          fineTuningLayers.push(layer);
          layer.trainable = true;

          break;
        }
      }

    }

    console.log(truncatedBase.summary());
    console.log(fineTuningLayers);
    return {truncatedBase, fineTuningLayers};
  }

  buildNewHead(inputShape) 
  {
    //const newHead = tf.sequential();
    //const input = tf.input({shape: [14,14,128]});

    //const flat=tf.layers.flatten({inputShape,name: 'flat'}).apply(input);
    //const dropOut1=tf.layers.dropout({rate: 0.2,name: 'dropout1'}).apply(flat);
    //const dense1=tf.layers.dense({units: 200, activation: 'relu',name: 'dense1'}).apply(dropOut1);
    //const dropOut2=tf.layers.dropout({rate: 0.2,name: 'dropout2'}).apply(dense1);

    //const out_cls=tf.layers.dense({units: 1, activation: 'softmax', kernelInitializer:'zeros', name: 'cls_out'}).apply(dropOut2);
    //const out_rgr=tf.layers.dense({units: 4, activation: 'linear', kernelInitializer:'zeros', name: 'rgr_out'}).apply(dropOut2);

    //const newHead = tf.model({inputs: input, outputs: [ <tf.SymbolicTensor> out_cls, <tf.SymbolicTensor> out_rgr ], name: 'modelNewHead' });

    const newHead = tf.sequential();
    newHead.add(tf.layers.flatten({inputShape}));
    newHead.add(tf.layers.dropout({rate: 0.5,name: 'dropout1'}));
   
    
    //newHead.add(tf.layers.dense({units: 200, activation: 'relu'}));
    //newHead.add(tf.layers.dropout({rate: 0.2,name: 'dropout2'}));
    // Five output units:
    //   - The first is a shape indictor: predicts whether the target
    //     shape is a triangle or a rectangle.
    //   - The remaining four units are for bounding-box prediction:
    //     [left, right, top, bottom] in the unit of pixels.
    //newHead.add(tf.layers.dense({units: 5, activation: 'relu'}));
    newHead.add(tf.layers.dense({units: 5}));
    
    return newHead;
  }

  async buildObjectDetectionModel() 
  {
    const {truncatedBase, fineTuningLayers} = await this.loadTruncatedBase();
  
    // Build the new head model.
    const newHead = this.buildNewHead(truncatedBase.outputs[0].shape.slice(1));
    const newOutput = newHead.apply(truncatedBase.outputs[0]);
    const model = tf.model({inputs: truncatedBase.inputs, outputs: <tf.SymbolicTensor>newOutput, name: 'modelLast'});

    return {model, fineTuningLayers};
  }


  async  trainObjectDetectionModel(model,fineTuningLayers,images,targets) 
  {

    function onBatchEnd(batch, logs) 
    {
      console.log('Accuracy', logs.acc);
      console.log('CrossEntropy', logs.ce);
      console.log('All', logs);
    }
    
    //cls_targets.print();
    //rgr_targets.print();
    console.log(images.dataSync().length);
    //console.log(cls_targets.dataSync().length);
    //console.log(rgr_targets.dataSync().length);

    console.log('Phase 1 of 2: initial transfer learning');
    //await model.fit(images, {modelNewHead:[cls_targets,rgr_targets]}, {
    await model.fit(images, targets, 
      {
      epochs: 6,
      batchSize: 20,
      validationSplit: 0.2,
      callbacks: {onBatchEnd}
   
    }).then(info => {
      console.log
      console.log('Final accuracy', info.history.acc);
      console.log('Cross entropy', info.ce);
      console.log('All', info);
    });;

    for (const layer of fineTuningLayers) 
    {
      layer.trainable = true;
    }
    
    //model.compile({loss: this.customLossFunction , optimizer: tf.train.adam(5e-3), metrics: ['accuracy', 'crossentropy']});
    
  // Do fine-tuning.
  // The batch size is reduced to avoid CPU/GPU OOM. This has
  // to do with the unfreezing of the fine-tuning layers above,
  // which leads to higher memory consumption during backpropagation.
  
    //console.log('Phase 2 of 2: fine-tuning phase');

    //await model.fit(images, targets, {
    //  epochs: 4,
    //  batchSize: 6,
    //  validationSplit: 0.2,
    //  callbacks: {onBatchEnd}
    //}).then(info => {
    //  console.log('Final accuracy', info.history.acc);
    //  console.log('Cross entropy', info.ce);
    //  console.log('All', info);
    //});;

    const saveResults = await model.save('downloads://my-model-1');

    console.log('Model saved');
  }


  parseImages(batchSize)
  {
    let allTextLines = this.csvContent.split(/\r|\n|\r/);
    
    const csvSeparator = ',';
    const csvSeparator_2 = '.';
    
    for ( let i = 0; i < batchSize; i++) 
    {
      // split content based on comma
     
      const cols: string[] = allTextLines[i].split(csvSeparator);
      

      if (cols[0].split(csvSeparator_2)[1]=="jpg") 
      {  
        this.src.push("../assets/images/"+ cols[0])
        
        if (cols[3]=="mouse") 
        { 
          this.label.push(Number('1'));
        } 

        if (cols[3]=="remote") 
        { 
          this.label.push(Number('0'));
        } 

        //if (cols[3]=="remote") 
        //{ 
        //  this.label.push(Number('0'));
        //} 

        //this.xmin.push(  (Number(cols[4])-112)/112  );
        //this.xmax.push(  (Number(cols[5])-112)/112  );
        //this.ymin.push(  (Number(cols[6])-112)/112  );
        //this.ymax.push(  (Number(cols[7])-112)/112  );

        this.xmin.push(  (Number(cols[4])) );
        this.xmax.push(  (Number(cols[5])) );
        this.ymin.push(  (Number(cols[6])) );
        this.ymax.push(  (Number(cols[7])) );

      } 
    }
  }

  onFileLoad(fileLoadedEvent) 
  {
    console.log('onFileLoad start');
    const textFromFileLoaded = fileLoadedEvent.target.result;              
    this.csvContent = textFromFileLoaded;  
  }

  onFileSelect(input: HTMLInputElement) 
  {
    console.log('onFileSelect start');

    const files = input.files;
    var content = this.csvContent;    
  
    if (files && files.length) 
    {
      console.log("Filename: " + files[0].name);
      console.log("Type: " + files[0].type);
      console.log("Size: " + files[0].size + " bytes");

      const fileToRead = files[0];

      const fileReader: FileReader = new FileReader();
      fileReader.onload = (event: Event) => {     
        const textFromFileLoaded = fileReader.result;              
        this.csvContent = textFromFileLoaded;   }

      fileReader.readAsText(fileToRead, "UTF-8");
    }
  }

  capture(imgId) 
  {
    // Reads the image as a Tensor from the webcam <video> element.
    
    this.picture = <HTMLImageElement> document.getElementById(imgId);
      
    const trainImage = tf.browser.fromPixels(this.picture);
    const trainim = trainImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));

    // Expand the outer most dimension so we have a batch size of 1.
    //const batchedImage = trainImage.expandDims(0);

    // Normalize the image between -1 and 1. The image comes in between 0-255,
    // so we divide by 127 and subtract 1.

    return trainim;
  }

  async train()
  {
    this.parseImages(120);
    const {model, fineTuningLayers} = await this.buildObjectDetectionModel();
    const {images, targets} = this.generateData(this.csvContent,120);
    
    model.compile({loss: this.customLossFunction, optimizer: tf.train.adam(1e-3), metrics: ['accuracy','crossentropy']});

    this.trainObjectDetectionModel(model,fineTuningLayers,images,targets);
  }

  wait(ms)
  {
    var start = new Date().getTime();
    var end = start;
    while(end < start + ms) 
    {
      end = new Date().getTime();
    }
  }

  generateData (trainData,batchSize)
  {
    const imageTensors = [];
    const targetTensors = [];

    const cls_targetTensors = [];
    const rgr_targetTensors = [];

    console.log(trainData)

    let allTextLines = this.csvContent.split(/\r|\n|\r/);
    
    const csvSeparator = ',';
    const csvSeparator_2 = '.';
    
    for ( let i = 0; i < batchSize; i++) 
    {
      // split content based on comma
      
      const cols: string[] = allTextLines[i].split(csvSeparator);
      console.log(cols[0].split(csvSeparator_2)[0])

      if (cols[0].split(csvSeparator_2)[1]=="jpg") 
      {
        
        const imageTensor = this.capture(i);
        
        const cls_targetTensor = tf.tensor1d([this.label[i]]);
        const rgr_targetTensor =tf.tensor1d([this.xmin[i],this.xmax[i],this.ymin[i],this.ymax[i]]);
        const targetTensor =tf.tensor1d([this.label[i],this.xmin[i],this.xmax[i],this.ymin[i],this.ymax[i]]);
        
        //const targetTensor=tf.tensor1d([this.label[i]]);
        

        imageTensors.push(imageTensor);
        targetTensors.push(targetTensor);
        cls_targetTensors.push(cls_targetTensor);  
        rgr_targetTensors.push(rgr_targetTensor); 

        imageTensor.print();
        //cls_targetTensor.print();
        //rgr_targetTensor.print();

      //tf.dispose([image, targetTensor]);

      } 

    }
    const images = tf.stack(imageTensors);
    const targets = tf.stack(targetTensors);   
    //targets[0] = tf.stack(cls_targetTensors);
    //targets[1]=  tf.stack(rgr_targetTensors);
    //return {images, cls_targets, rgr_targets};
    return {images, targets};

  }


}
