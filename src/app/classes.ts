 /* 3131B Grigoras Emanuel, Ciprian Iacob, Iriciuc Andrei, Ilciuc Sergiu */

export interface ObjectDetectionClass {
  name: string;
  id: number;
  displayName: string;
}

export const CLASSES: { [key: string]: ObjectDetectionClass } = {
  1: {
    name: '1leu',
    id: 1,
    displayName: '1leu',
  },
  2: {
    name: '5lei',
    id: 2,
    displayName: '5lei',
  },
};
