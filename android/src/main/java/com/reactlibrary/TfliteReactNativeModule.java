
package com.reactlibrary;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Canvas;
import android.util.Base64;
import android.util.Log;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Callback;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Vector;

public class TfliteReactNativeModule extends ReactContextBaseJavaModule {

  private final ReactApplicationContext reactContext;
  private Interpreter tfLite;
  private int inputSize = 0;
  private Vector<String> labels;
  float[][] labelProb;
  private static final int BYTES_PER_CHANNEL = 4;


  public TfliteReactNativeModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.reactContext = reactContext;
  }

  @Override
  public String getName() {
    return "TfliteReactNative";
  }

  @ReactMethod
  private void loadModel(final String modelPath, final String labelsPath, final int numThreads, final Callback callback)
      throws IOException {
    AssetManager assetManager = reactContext.getAssets();
    AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

    final Interpreter.Options tfliteOptions = new Interpreter.Options();
    tfliteOptions.setNumThreads(numThreads);
    tfLite = new Interpreter(buffer, tfliteOptions);

    if (labelsPath.length() > 0) {
      loadLabels(assetManager, labelsPath);
    }

    callback.invoke(null, "success");
  }

  private void loadLabels(AssetManager assetManager, String path) {
    BufferedReader br;
    try {
      br = new BufferedReader(new InputStreamReader(assetManager.open(path)));
      String line;
      labels = new Vector<>();
      while ((line = br.readLine()) != null) {
        labels.add(line);
      }
      labelProb = new float[1][labels.size()];
      br.close();
    } catch (IOException e) {
      throw new RuntimeException("Failed to read label file", e);
    }
  }


  private WritableArray GetTopN(int numResults, float threshold) {
    PriorityQueue<WritableMap> pq =
        new PriorityQueue<>(
            1,
            new Comparator<WritableMap>() {
              @Override
              public int compare(WritableMap lhs, WritableMap rhs) {
                return Double.compare(rhs.getDouble("confidence"), lhs.getDouble("confidence"));
              }
            });
    final float[] classes = new float[labels.size()];
    for (int c = 0; c < labels.size(); c++) {
        classes[c] = labelProb[0][c];
    }
    softmax(classes);

    for (int i = 0; i < labels.size(); ++i) {
      float confidence = classes[i];//labelProb[0][i];
      if (confidence > threshold) {
        WritableMap res = Arguments.createMap();
        res.putInt("index", i);
        res.putString("label", labels.size() > i ? labels.get(i) : "unknown");
        res.putDouble("confidence", confidence);
        pq.add(res);
      }
    }

    WritableArray results = Arguments.createArray();
    int recognitionsSize = Math.min(pq.size(), numResults);
    for (int i = 0; i < recognitionsSize; ++i) {
      results.pushMap(pq.poll());
    }
    return results;
  }


  private WritableArray GetEmbedding() {
    WritableArray results = Arguments.createArray();
    for (int i = 0; i < labels.size(); ++i) {
      float confidence = labelProb[0][i];
      WritableMap res = Arguments.createMap();
      res.putInt("index", i);
      res.putDouble("value", confidence);
      results.pushMap(res);
    }

    Log.d("TfLite-Lib", "" + results.size());
    Log.d("TfLite-Lib", "" + results.toString());

    return results;
  }
    ByteBuffer feedInputTensorImage(String path, float mean1, float mean2, float mean3, float std1, float std2, float std3) throws IOException {
    Tensor tensor = tfLite.getInputTensor(0);
    inputSize = tensor.shape()[1];
    int inputChannels = tensor.shape()[3];

    InputStream inputStream = new FileInputStream(path.replace("file://", ""));
    Bitmap bitmapRaw = BitmapFactory.decodeStream(inputStream);
    int[] intValues = new int[inputSize * inputSize];
    int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : BYTES_PER_CHANNEL;
    ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
    imgData.order(ByteOrder.nativeOrder());

    Bitmap bitmap = Bitmap.createScaledBitmap(bitmapRaw, inputSize, inputSize, true);
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());


    int pixel = 0;
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValues[pixel++];
        if (tensor.dataType() == DataType.FLOAT32) {
          imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean1) / std1);
          imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean2) / std2);
          imgData.putFloat(((pixelValue & 0xFF) - mean3) / std3);
        } else {
          imgData.put((byte) ((pixelValue >> 16) & 0xFF));
          imgData.put((byte) ((pixelValue >> 8) & 0xFF));
          imgData.put((byte) (pixelValue & 0xFF));
        }
      }
    }

    return imgData;
  }

  @ReactMethod
  private void runModelOnImage(final String path,
          final float mean1,
          final float mean2,
          final float mean3,
          final float std1,
          final float std2,
          final float std3,
          final int numResults,
                               final float threshold, final Callback callback) throws IOException {

    tfLite.run(feedInputTensorImage(path, mean1, mean2, mean3, std1, std2, std3), labelProb);

    callback.invoke(null, GetTopN(numResults, threshold)); }

  @ReactMethod
  private void close() {
    tfLite.close();
    labels = null;
    labelProb = null;
  }


  private void softmax(final float[] vals) {
    float max = Float.NEGATIVE_INFINITY;
    for (final float val : vals) {
      max = Math.max(max, val);
    }
    float sum = 0.0f;
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = (float) Math.exp(vals[i] - max);
      sum += vals[i];
    }
    for (int i = 0; i < vals.length; ++i) {
      vals[i] = vals[i] / sum;
    }
  }

}
