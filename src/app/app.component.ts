/* 3131B Grigoras Emanuel, Ciprian Iacob, Iriciuc Andrei, Ilciuc Sergiu */

import { Component, OnInit } from '@angular/core';
import { ObjectDetection } from './ObjectDetection';

//import COCO-SSD model as cocoSSD
import * as cocoSSD from '@tensorflow-models/coco-ssd';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent implements OnInit {
  private video: HTMLVideoElement;
  public firstDraw = false;
  public changedModel = false;
  public scoreThreshold = 0.9;
  public modelType="Custom";

  public detectedObjects = [];
  public detectedObjectsSet = {};

  ngOnInit() {
    this.initialize_camera();
    this.predictWithCocoModel();
  }

  loadModel(custom) {
    this.modelType = custom? "Custom": "Official";
    this.changedModel = true;

    setTimeout(() => {
      this.changedModel = false;
      this.predictWithCocoModel(custom);
    }, 100);
  }

  public async predictWithCocoModel(custom = true) {

    let model = null;

    if (custom) {
      const x = new ObjectDetection('lite_mobilenet_v2');
      const modelPromise = load();
      model = await modelPromise;
    }
    else {
      model = await cocoSSD.load({ base: 'lite_mobilenet_v2' });
    }

    this.detectFrame(this.video, model);
    console.log('model loaded');
  }

  initialize_camera() {
    this.video = <HTMLVideoElement>document.getElementById("vid");

    navigator.mediaDevices
      .getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
        }
      })
      .then(stream => {
        this.video.srcObject = stream;
        this.video.onloadedmetadata = () => {
          this.video.play();
        };
      });
  }

  detectFrame = (video, model) => {
    if (this.changedModel) {
      return;
    }

    model.detect(video).then(predictions => {
      this.renderPredictions(predictions);
      requestAnimationFrame(() => {
        this.detectFrame(video, model);
      });
    });
  }

  renderPredictions = predictions => {
    this.firstDraw = true;

    const canvas = <HTMLCanvasElement>document.getElementById("canvas");

    const ctx = canvas.getContext("2d");

    canvas.width = 600;
    canvas.height = 600;

    const rateW = canvas.width / 300;
    const rateH = canvas.height / 300;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";
    ctx.drawImage(this.video, 0, 0, canvas.width, canvas.height);


    predictions.forEach(prediction => {
      if (prediction.score > this.scoreThreshold) {
        const x = prediction.bbox[0] * rateW;
        const y = prediction.bbox[1] * rateH;
        const width = prediction.bbox[2] * rateW;
        const height = prediction.bbox[3] * rateH;

        // Draw the bounding box.
        ctx.strokeStyle = "#00FFFF";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, width, height);

        // Draw the label background.
        ctx.fillStyle = "#00FFFF";
        const textWidth = ctx.measureText(prediction.class + '100%').width;
        const textHeight = parseInt(font, 10); // base 10
        ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
      }
    });

    predictions.forEach(prediction => {
      if (prediction.score > this.scoreThreshold) {
        const x = prediction.bbox[0] * rateW;
        const y = prediction.bbox[1] * rateH;

        // Draw the text last to ensure it's on top.
        ctx.fillStyle = "#000000";
        ctx.fillText(prediction.class + ' ' + (prediction.score * 100).toFixed() + '%', x, y);

        if (!this.detectedObjectsSet[prediction.class]) {
          this.detectedObjectsSet[prediction.class] = prediction.class;
          this.detectedObjects.push(prediction.class);
        }
      }
    });
  };
}

export async function load() {
  const objectDetection = new ObjectDetection('lite_mobilenet_v2');
  await objectDetection.load();
  return objectDetection;
}
