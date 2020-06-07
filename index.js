import { NativeModules, Image } from "react-native";

const { TfliteReactNative } = NativeModules;

class Tflite {
  loadModel(args, callback) {
    TfliteReactNative.loadModel(
      args["model"],
      args["labels"] || "",
      args["numThreads"] || 1,
      (error, response) => {
        callback && callback(error, response);
      }
    );
  }

  runModelOnImage(args, callback) {
    TfliteReactNative.runModelOnImage(
      args["path"],
      args["imageMean1"],
      args["imageMean2"],
      args["imageMean3"],
      args["imageStd1"],
      args["imageStd2"],
      args["imageStd3"],
      args["numResults"] || 10,
      args["threshold"] != null ? args["threshold"] : 0.1,
      (error, response) => {
        callback && callback(error, response);
      }
    );
  }

  close() {
    TfliteReactNative.close();
  }
}

export default Tflite;
