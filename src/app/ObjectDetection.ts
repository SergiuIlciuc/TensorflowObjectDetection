// Continutul acestui fisier este preluat de la adresa aceasta https://developer.aliyun.com/mirror/npm/package/@tensorflow-models/coco-ssd/v/0.1.0

import * as tfconv from '@tensorflow/tfjs-converter';

import * as tf from '@tensorflow/tfjs-core';
import { CLASSES } from './classes';

export type ObjectDetectionBaseModel =
'mobilenet_v1' | 'mobilenet_v2' | 'lite_mobilenet_v2';

export interface DetectedObject {
bbox: [number, number, number, number];  // [x, y, width, height]
class: string;
score: number;
}

export interface ModelConfig {
base?: ObjectDetectionBaseModel;
modelUrl?: string;
}

  /**
   * Dispose the tensors allocated by the model. You should call this when you
   * are done with the model.
   */
export class ObjectDetection {
  private modelPath: string;
  private model: tfconv.GraphModel;

  constructor(base: ObjectDetectionBaseModel, modelUrl?: string) {
    this.modelPath = '../../assets/model.json';
  }

  async load() {
    this.model = await tfconv.loadGraphModel(this.modelPath);
    // Warmup the model.
    const result = await this.model.executeAsync(tf.zeros([1, 300, 300, 3])) as
      tf.Tensor[];
    await Promise.all(result.map(t => t.data()));
    result.map(t => t.dispose());
  }

  /**
   * Infers through the model.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   * @param maxNumBoxes The maximum number of bounding boxes of detected
   * objects. There can be multiple objects of the same class, but at different
   * locations. Defaults to 20.
   */
  private async infer(
    img: tf.Tensor3D | ImageData | HTMLImageElement | HTMLCanvasElement |
      HTMLVideoElement,
    maxNumBoxes: number): Promise<DetectedObject[]> {
    const batched = tf.tidy(() => {
      if (!(img instanceof tf.Tensor)) {
        img = tf.browser.fromPixels(img);
      }
      // Reshape to a single-element batch so we can pass it to executeAsync.
      return img.expandDims(0);
    });
    const height = batched.shape[1];
    const width = batched.shape[2];

    // model returns two tensors:
    // 1. box classification score with shape of [1, 1917, 90]
    // 2. box location with shape of [1, 1917, 1, 4]
    // where 1917 is the number of box detectors, 90 is the number of classes.
    // and 4 is the four coordinates of the box.
    const result = await this.model.executeAsync(batched) as tf.Tensor[];
    // console.log(result);

    const scores = result[0].dataSync() as Float32Array;
    const boxes = result[1].dataSync() as Float32Array;
    // console.log('scores', result[0]);
    // console.log('scores1', result[1]);

    // clean the webgl tensors
    batched.dispose();
    tf.dispose(result);

    const [maxScores, classes] =
      this.calculateMaxScores(scores, result[0].shape[1], result[0].shape[2]);

    const prevBackend = tf.getBackend();
    // run post process in cpu
    tf.setBackend('cpu');
    // console.log('BOXES', result[1]);
    const indexTensor = tf.tidy(() => {
      const boxes2 =
        tf.tensor2d(boxes, [result[1].shape[1], 4]);
      return tf.image.nonMaxSuppression(
        boxes2, maxScores, maxNumBoxes, 0.5, 0.5);
    });

    const indexes = indexTensor.dataSync() as Float32Array;
    indexTensor.dispose();

    // restore previous backend
    tf.setBackend(prevBackend);

    return this.buildDetectedObjects(
      width, height, boxes, maxScores, indexes, classes);
  }

  private buildDetectedObjects(
    width: number, height: number, boxes: Float32Array, scores: number[],
    indexes: Float32Array, classes: number[]): DetectedObject[] {
    const count = indexes.length;
    const objects: DetectedObject[] = [];
    for (let i = 0; i < count; i++) {
      const bbox = [];
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j];
      }
      const minY = bbox[0] * height;
      const minX = bbox[1] * width;
      const maxY = bbox[2] * height;
      const maxX = bbox[3] * width;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;
      objects.push({
        bbox: bbox as [number, number, number, number],
        class: CLASSES[classes[indexes[i]] + 1].displayName,
        score: scores[indexes[i]]
      });
    }
    return objects;
  }

  private calculateMaxScores(
    scores: Float32Array, numBoxes: number,
    numClasses: number): [number[], number[]] {
    const maxes = [];
    const classes = [];
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE;
      let index = -1;
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j];
          index = j;
        }
      }
      maxes[i] = max;
      classes[i] = index;
    }
    return [maxes, classes];
  }

  async detect(
    img: tf.Tensor3D | ImageData | HTMLImageElement | HTMLCanvasElement |
      HTMLVideoElement,
    maxNumBoxes = 20): Promise<DetectedObject[]> {
    return this.infer(img, maxNumBoxes);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }
}
