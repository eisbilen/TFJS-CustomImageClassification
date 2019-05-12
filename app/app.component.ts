//-------------------------------------------------------------
// importes tfjs and Angular Material classes
//-------------------------------------------------------------
import {Component, OnInit, ViewChild , ViewEncapsulation} from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import {categoricalCrossentropy} from '@tensorflow/tfjs-layers/dist/exports_metrics';
import {MatTable, MatPaginator, MatTableDataSource} from '@angular/material';
import {MatSnackBar} from '@angular/material';
//-------------------------------------------------------------
// defines 'TrainingImageList' interface to store training dataset
//-------------------------------------------------------------
export interface TrainingImageList 
{
  ImageSrc: string; // this is to get the
  LabelX1: number; // 
  LabelX2: number;
  Class: string;
};
//-------------------------------------------------------------
// defines 'TrainingMetrics' interface to store training metrics
//-------------------------------------------------------------
export interface TrainingMetrics 
{
  acc: number; // training accuracy value 
  ce: number; // cross entropy value 
  loss: number; // loss function value 
};
//-------------------------------------------------------------
@Component
({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
  encapsulation: ViewEncapsulation.None
})
//-------------------------------------------------------------
// 'AppComponent' class
//-------------------------------------------------------------
export class AppComponent implements OnInit 
{
  private isImagesListed: Boolean;
  private isImagesListPerformed: Boolean;
  
  constructor(private snackBar: MatSnackBar) {}

  private picture: HTMLImageElement;
  
  public tableRows: TrainingImageList[]=[]; //instance of TrainingImageList 
  public dataSource = new MatTableDataSource<TrainingImageList>(this.tableRows);  //datasourse as
  public traningMetrics: TrainingMetrics[]=[]; //instance of TrainingMetrics

  public displayedColumns: string[] = ['ImageSrc','Class','Label_X1','Label_X2'];
  
  private csvContent: any;

  private label_x1: number[]=[];
  private label_x2: number[]=[];

  public ProgressBarValue: number;
  
  @ViewChild(MatTable) table: MatTable<any>;
  @ViewChild(MatPaginator) paginator: MatPaginator;
 
  ngOnInit()
  {   
    this.isImagesListed=false;
    this.isImagesListPerformed=false;
    this.ProgressBarValue=0;
  }
  //-------------------------------------------------------------
  // calls generateData() to prepare the training dataset
  // calls getModfiedMobilenet() to prepare the model for training
  // calls fineTuneModifiedModel() to finetune the model
  //-------------------------------------------------------------
  async train()
  { 
    const {images, targets} = this.generateData(this.csvContent,120);
    this.ProgressBarValue=35;
    this.openSnackBar("Images are loaded into the memory as tensor !","Close");

    const mobilenetModified = await this.getModifiedMobilenet();
    this.ProgressBarValue=50;
    this.openSnackBar("Modefiled Mobilenet AI Model is loaded !","Close");

    await this.fineTuneModifiedModel(mobilenetModified,images,targets);
    this.openSnackBar("Model training is completed !","Close");
    this.ProgressBarValue=100;
  }
  //-------------------------------------------------------------
  // calls parseImages() to populate imageSrc and targets as a list 
  // 
  //-------------------------------------------------------------
  async loadCSV()
  { 
    this.parseImages(120);

    if (this.isImagesListPerformed)
    {
      this.openSnackBar("Training images are listed !","Close");
    }
    if (!this.isImagesListPerformed)
    {
      this.openSnackBar("Please reset the dataset to upload new CSV file !","Reset");
    }
  }
  reset()
  {
  
  };
  //-------------------------------------------------------------
  // modifies the pre-trained mobilenet to detect malaria infected
  // cells, freezes layers to train only the last couple of layers
  //-------------------------------------------------------------
  async getModifiedMobilenet()
  {
    const trainableLayers = ['denseModified','conv_pw_13_bn','conv_pw_13','conv_dw_13_bn','conv_dw_13'];
    const mobilenet =  await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    console.log('Mobilenet model is loaded')

    const x=mobilenet.getLayer('global_average_pooling2d_1');
    const predictions= <tf.SymbolicTensor> tf.layers.dense({units: 2, activation: 'softmax',name: 'denseModified'}).apply(x.output);
    let mobilenetModified = tf.model({inputs: mobilenet.input, outputs: predictions, name: 'modelModified' });
    console.log('Mobilenet model is modified')
  
    mobilenetModified = this.freezeModelLayers(trainableLayers,mobilenetModified)
    console.log('ModifiedMobilenet model layers are freezed')

    mobilenetModified.compile({loss: categoricalCrossentropy, optimizer: tf.train.adam(1e-3), metrics: ['accuracy','crossentropy']});


    return mobilenetModified 
  }
  //-------------------------------------------------------------
  // freezes mobilenet layers to make them untrainable
  // just keeps final layers trainable with argument trainableLayers
  //-------------------------------------------------------------
  freezeModelLayers(trainableLayers,mobilenetModified)
  {
    for (const layer of mobilenetModified.layers) 
    {
      layer.trainable = false;
      for (const tobeTrained of trainableLayers) 
      {
        if (layer.name.indexOf(tobeTrained) === 0) 
        {
          layer.trainable = true;
          break;
        }
      }
    }
    return mobilenetModified;
  }
  //-------------------------------------------------------------
  // finetunes the modified mobilenet model in 5 training batches
  // takes model, images and targets as arguments
  //-------------------------------------------------------------
  async fineTuneModifiedModel(model,images,targets)
  {
    function onBatchEnd(batch, logs) 
    {
      console.log('Accuracy', logs.acc);
      console.log('CrossEntropy', logs.ce);
      console.log('All', logs);
    }
    console.log('Finetuning the model...');

    await model.fit(images, targets, 
    {
      epochs: 5,
      batchSize: 24,
      validationSplit: 0.2,
      callbacks: {onBatchEnd}
   
    }).then(info => {
      console.log
      console.log('Final accuracy', info.history.acc);
      console.log('Cross entropy', info.ce);
      console.log('All', info);
      console.log('All', info.history['acc'][0]);
      
      for ( let k = 0; k < 5; k++) 
    {
      this.traningMetrics.push({acc: 0, ce: 0 , loss: 0});

      this.traningMetrics[k].acc=info.history['acc'][k];
      this.traningMetrics[k].ce=info.history['ce'][k];
      this.traningMetrics[k].loss=info.history['loss'][k]; 
    }
      images.dispose();
      targets.dispose();
      model.dispose();
    });;
  
  }
  //-------------------------------------------------------------
  // stores Image Src and Class info in CSV file
  // populates the MatTable rows and paginator
  // populates the targets as [1,0] uninfected, [0,1] parasitized
  //-------------------------------------------------------------
  parseImages(batchSize)
  {
    if (this.isImagesListed) 
    {
      this.isImagesListPerformed=false;
      return;
    }

    let allTextLines = this.csvContent.split(/\r|\n|\r/);
    
    const csvSeparator = ',';
    const csvSeparator_2 = '.';
    
    for ( let i = 0; i < batchSize; i++) 
    {
      // split content based on comma
      const cols: string[] = allTextLines[i].split(csvSeparator);
      
      this.tableRows.push({ImageSrc: '', LabelX1: 0 , LabelX2: 0, Class: ''});

      if (cols[0].split(csvSeparator_2)[1]=="png") 
      {  
        
        if (cols[1]=="Uninfected") 
        { 
          this.label_x1.push(Number('1'));
          this.label_x2.push(Number('0'));

          this.tableRows[i].ImageSrc="../assets/"+ cols[0];
          this.tableRows[i].LabelX1=1;
          this.tableRows[i].LabelX2=0;
          this.tableRows[i].Class="Uninfected";
        } 

        if (cols[1]=="Parasitized") 
        { 
          this.label_x1.push(Number('0'));
          this.label_x2.push(Number('1'));
      
          this.tableRows[i].ImageSrc="../assets/"+ cols[0];
          this.tableRows[i].LabelX1=0;
          this.tableRows[i].LabelX2=1;
          this.tableRows[i].Class="Parasitized";
        } 

      } 
    }
    this.table.renderRows();
    this.dataSource.paginator = this.paginator;
    
    this.isImagesListed=true;
    this.isImagesListPerformed=true;
  }

  //-------------------------------------------------------------
  // this function generate input and target tensors for the training
  // input tensor is produced from 224x224x3 image in HTMLImageElement
  // target tensor shape2 is produced from the class definition
  //-------------------------------------------------------------
  generateData (trainData,batchSize)
  {
    const imageTensors = [];
    const targetTensors = [];

    let allTextLines = this.csvContent.split(/\r|\n|\r/);
    
    const csvSeparator = ',';
    const csvSeparator_2 = '.';
    
    for ( let i = 0; i < batchSize; i++) 
    {
      // split content based on comma
      const cols: string[] = allTextLines[i].split(csvSeparator);
      console.log(cols[0].split(csvSeparator_2)[0])

      if (cols[0].split(csvSeparator_2)[1]=="png") 
      {
        console.log(i)
        const imageTensor = this.capture(i);
        let targetTensor =tf.tensor1d([this.label_x1[i],this.label_x2[i]]);

        targetTensor.print();
        imageTensors.push(imageTensor);
        targetTensors.push(targetTensor);
  
        imageTensor.print(true);
      } 
    }
    const images = tf.stack(imageTensors);
    const targets = tf.stack(targetTensors);   

    return {images, targets};
  }
  //-------------------------------------------------------------
  // converts images in HTMLImageElement into the tensors
  // takes Image In in HTML as argument
  //-------------------------------------------------------------
  capture(imgId) 
  {
    // Reads the image as a Tensor from the <image> element.
    this.picture = <HTMLImageElement> document.getElementById(imgId);
    const trainImage = tf.browser.fromPixels(this.picture);

    // Normalize the image between -1 and 1. The image comes in between 0-255,
    // so we divide by 127 and subtract 1.
    const trainim = trainImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));

    return trainim;
  }
  //-------------------------------------------------------------
  // onFileLoad and onFileSelect functions opens file browser 
  // and stores the selected CVS file content in csvContent variable
  //-------------------------------------------------------------
  onFileLoad(fileLoadedEvent) 
  {
    const textFromFileLoaded = fileLoadedEvent.target.result;              
    this.csvContent = textFromFileLoaded;  
  }
  onFileSelect(input: HTMLInputElement) 
  {
    const files = input.files;
    
    if (files && files.length) 
    {
      const fileToRead = files[0];

      const fileReader: FileReader = new FileReader();
      fileReader.onload = (event: Event) => {     
        const textFromFileLoaded = fileReader.result;              
        this.csvContent = textFromFileLoaded;   }

      fileReader.readAsText(fileToRead, "UTF-8");

      console.log("Filename: " + files[0].name);
      console.log("Type: " + files[0].type);
      console.log("Size: " + files[0].size + " bytes");
    }
  }
  //-------------------------------------------------------------
  // defines 'getTotalUninfected','getTotalParatisized' functions 
  // to count uninfected and paratisized cell images
  //-------------------------------------------------------------
  getTotalUninfected() 
  {
    return this.tableRows.map(t => t.LabelX1).reduce((acc, value) => acc + value, 0);
  };
  getTotalPAratisized() 
  {
    return this.tableRows.map(t => t.LabelX2).reduce((acc, value) => acc + value, 0);
  };
  //-------------------------------------------------------------
  // call snack bar to inform the progress of training
  //-------------------------------------------------------------
  openSnackBar(message: string, action: string) 
  {
    this.snackBar.open(message, action, {duration: 4000});
  }

}
