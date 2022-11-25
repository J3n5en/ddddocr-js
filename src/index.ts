import path from "node:path";
import fs from "node:fs";

import Jimp from "jimp";
import { Tensor, InferenceSession } from "onnxruntime-node";

type Options = {
  onnxPath?: string;
  charsetsPath?: string;
};

class DdddOcr {
  #charsets: string[];
  #session: InferenceSession;

  private constructor(session: InferenceSession, charsets: string[]) {
    this.#session = session;
    this.#charsets = charsets;
  }

  static async create(options?: Options) {
    const charsetsPath =
      options?.charsetsPath || path.resolve(__dirname, "../onnx/charsets.json");

    const charsets = JSON.parse(
      fs.readFileSync(charsetsPath, { encoding: "utf-8" })
    );

    const session = await InferenceSession.create(
      options?.onnxPath || path.resolve(__dirname, "../onnx/common.onnx")
    );

    return new DdddOcr(session, charsets);
  }

  async classification(buff: Buffer) {
    const { image, dims } = await this.loadImage(buff);
    const inputTensor = this.coverImageToTensor(image, dims);
    const {
      output: { data: outputData },
    } = await this.#session.run({ input1: inputTensor });

    return [...outputData]
      .filter(Boolean)
      .map((i) => this.#charsets[Number(i)])
      .join("");
  }

  private async loadImage(buffer: Buffer) {
    return Jimp.read(buffer).then((imageBuffer) => {
      var width = imageBuffer.bitmap.width;
      var height = imageBuffer.bitmap.height;
      const dims = [1, 1, 64, Math.floor(width * (64 / height))];
      return {
        image: imageBuffer.resize(dims[3], dims[2]).grayscale(),
        dims,
      };
    });
  }

  private coverImageToTensor(image: Jimp, dims: number[]) {
    const redArray: number[] = [];
    const greenArray: number[] = [];
    const blueArray: number[] = [];
    for (let i = 0; i < image.bitmap.data.length; i += 4) {
      redArray.push(image.bitmap.data[i]);
      greenArray.push(image.bitmap.data[i + 1]);
      blueArray.push(image.bitmap.data[i + 2]);
    }

    const transposedData = redArray.concat(greenArray).concat(blueArray);

    const float32Data = new Float32Array(dims.reduce((a, b) => a * b));
    for (let i = 0; i < transposedData.length; i++) {
      float32Data[i] = transposedData[i] / 255.0;
    }

    return new Tensor("float32", float32Data, dims);
  }
}

export default DdddOcr;
