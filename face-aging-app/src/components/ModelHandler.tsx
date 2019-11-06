import React from "react";
import * as tf from "@tensorflow/tfjs";
import { async } from "q";

export default function ModelHandler() {
  const loadModel = async () => {
    const model = await tf.loadLayersModel(
      "https://foo.bar/tfjs_artifacts/model.json"
    );
  };

  return <div>Hei</div>;
}
