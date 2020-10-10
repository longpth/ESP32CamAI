/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.p4f.esp32camai.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import android.util.Log;

import com.p4f.esp32camai.MyConstants;

public class TFLiteObjectDetectionAPIModel extends Classifier {

  final static String TAG = "TFLiteObjectDetectionAPIModel";

  /** An immutable result returned by a Classifier describing what was recognized. */
  public class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    /** Time execution for detect the recognition by running the tflite*/
    private long timeExecution;

    public Recognition(
            final String id, final String title, final Float confidence, final RectF location, final long timeEx) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
      this.timeExecution = timeEx;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    public long getTimeExecution(){
      return this.timeExecution;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  public Vector<String> getLabels(){
    return labels;
  }

  public static float sigmoid(float x) {
    return (float)(1/( 1 + Math.pow(Math.E,(-1*x))));
  }

  /** Memory-map the model file in Assets. */
  private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
          throws IOException {
    AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  class BoxInfo{
    public BoxInfo(float xmin, float ymin, float xmax, float ymax, float score, String name){
      _xmin = xmin;
      _ymin = ymin;
      _xmax = xmax;
      _ymax = ymax;
      _score = score;
      _name = name;
    }
    public float _xmin;
    public float _ymin;
    public float _xmax;
    public float _ymax;
    public float _score;
    public String _name;
  }

  class MAX_SCORE{
    public MAX_SCORE(float score, int indx){
      this.score = score;
      this.indx = indx;
    }
    public float score;
    public int indx;
  };

  /** Optional GPU delegate for accleration. */
  private GpuDelegate gpuDelegateYolo = null;

  // Only return this many results.
  private static final int NUM_DETECTIONS = 10;
  // Float model
  private static final float IMAGE_STD = 255.0f;
  // Number of threads in the java app
  private static final int NUM_THREADS = 4;

  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptionsYolo = new Interpreter.Options();

  //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>for YOLOv3-tiny<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  private int[] intValuesYolo;
  private int inputSize;
  private float[][][][] outputRaw0;
  private float[][][][] outputRaw1;

  private ByteBuffer imgDataYolo;

  private Interpreter tfLiteYolo;

  private final float[] ANCHOR_BOXES = new float[] {81,82,  135,169,  344,319, 10,14,  23,27,  37,58};
  private final int[] GRID_SIZES = new int[] {13,26};
  private float scoreThreshold;

  private final int NUM_CLASSES = 80;

  private List<Recognition> resultsYolo = new ArrayList<>();

  MyConstants.MODEL_TYPE modelType;

  private TFLiteObjectDetectionAPIModel() {}

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @param modelFilename The filepath of the model GraphDef protocol buffer.
   * @param labelFilename The filepath of label file for classes.
   * @param inputSize The size of image input for YOLOv3
   */
  public static Classifier create(
      final AssetManager assetManager,
      final String modelFilename,
      final String labelFilename,
      final int inputSize,
      final Device device,
      final MyConstants.MODEL_TYPE modelType,
      final float scoreThreshold,
      final int queueSize,
      final int width,
      final int height)
      throws IOException {
    final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();

    InputStream labelsInput = null;
    String actualFilename = labelFilename.split("file:///android_asset/")[1];
    labelsInput = assetManager.open(actualFilename);
    BufferedReader br = null;
    br = new BufferedReader(new InputStreamReader(labelsInput));
    String line;
    while ((line = br.readLine()) != null) {
      Log.d(TAG, line);
      d.labels.add(line);
    }
    br.close();

    d.modelType = modelType;

    final String strModelYolo = modelFilename;

    switch (device) {
      case NNAPI:
        d.tfliteOptionsYolo.setUseNNAPI(true);
        break;
      case GPU:
        d.gpuDelegateYolo = new GpuDelegate();
        d.tfliteOptionsYolo.addDelegate(d.gpuDelegateYolo);
        Log.d("GPU delegated for YOLOv3-tiny","");
        break;
      case CPU:
        break;
    }

    d.tfliteOptionsYolo.setNumThreads(NUM_THREADS);

    try {
      d.tfLiteYolo = new Interpreter(loadModelFile(assetManager, strModelYolo), d.tfliteOptionsYolo);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }

    // Pre-allocate buffers.
    int numBytesPerChannelYolo = 4; // Floating point

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>YOLOv3-tiny<<<<<<<<<<<<<<<<<<<<<<<<<<
    d.inputSize = inputSize;
    d.imgDataYolo = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannelYolo);
    d.imgDataYolo.order(ByteOrder.nativeOrder());
    d.intValuesYolo = new int[d.inputSize * d.inputSize];
    d.scoreThreshold = scoreThreshold;

    d.outputRaw0 = new float[1][13][13][255];
    d.outputRaw1 = new float[1][26][26][255];

    d.bmpQueue = new LinkedList<Bitmap>()
    {
      @Override
      public boolean add(Bitmap eldest)
      {
        if (this.size() >= queueSize){
          this.remove();
          return super.add(eldest);
        }else{
          return super.add(eldest);
        }
      }
    };

    d.threadDectection = new Thread(d);

    d.width = width;
    d.height = height;

    return d;
  }

  /**
   * Get object results from network
   *
   * @param results store the result.
   *
   * @return true(result is ready for reading)/false(result not ready for reading)
   */
  public void getResult(List<Recognition> results){
    synchronized (lockResult){
    }
  }

  @Override
  protected void processing(Bitmap bmp){
    resultsYolo = recognizeImageYolo(bmp);
  }

  @Deprecated @SuppressWarnings ( "FloatingPointEquality" )
  public static short toHalfFloat(final float v)
  {
    if(Float.isNaN(v)) throw new UnsupportedOperationException("NaN to half conversion not supported!");
    if(v == Float.POSITIVE_INFINITY) return(short)0x7c00;
    if(v == Float.NEGATIVE_INFINITY) return(short)0xfc00;
    if(v == 0.0f) return(short)0x0000;
    if(v == -0.0f) return(short)0x8000;
    if(v > 65504.0f) return 0x7bff;  // max value supported by half float
    if(v < -65504.0f) return(short)( 0x7bff | 0x8000 );
    if(v > 0.0f && v < 5.96046E-8f) return 0x0001;
    if(v < 0.0f && v > -5.96046E-8f) return(short)0x8001;

    final int f = Float.floatToIntBits(v);

    return(short)((( f>>16 ) & 0x8000 ) | (((( f & 0x7f800000 ) - 0x38000000 )>>13 ) & 0x7c00 ) | (( f>>13 ) & 0x03ff ));
  }

  // return the max score and class index
  private MAX_SCORE findScoreYolo(float[] array, int anchor_idx){
    int sizeOfOneBox = NUM_CLASSES+5;
    float max = sigmoid(array[anchor_idx*sizeOfOneBox+5]);
    float objectness = sigmoid(array[anchor_idx*sizeOfOneBox+4]);
    int indx = 0;
    for(int i = 5; i < sizeOfOneBox; i++)
    {
      float cur = sigmoid(array[i+anchor_idx*sizeOfOneBox]);
      if(cur > max)
      {
        max = cur;
        indx = i-5;
      }
    }
    MAX_SCORE score = new MAX_SCORE(max*objectness, indx);
    return score;
  }

  private List<Recognition> recognizeImageYolo(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");

    Bitmap resizeBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, false);

    resizeBitmap.getPixels(intValuesYolo, 0, resizeBitmap.getWidth(), 0, 0, resizeBitmap.getWidth(), resizeBitmap.getHeight());

    imgDataYolo.rewind();
    for (int i = 0; i < inputSize; ++i) {
      for (int j = 0; j < inputSize; ++j) {
        int pixelValue = intValuesYolo[i * inputSize + j];
          if (modelType== MyConstants.MODEL_TYPE.FLOAT32) {
              imgDataYolo.putFloat((float) (pixelValue & 0xFF) / IMAGE_STD);
              imgDataYolo.putFloat((float) ((pixelValue >> 8) & 0xFF) / IMAGE_STD);
              imgDataYolo.putFloat((float) ((pixelValue >> 16) & 0xFF) / IMAGE_STD);
          }else{
            //TODO
          }
      }
    }

    Trace.endSection(); // preprocessBitmap

    // Copy the input data into TensorFlow.
    Trace.beginSection("feed");

    final long startTime = SystemClock.uptimeMillis();

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>YOLOv3-tiny<<<<<<<<<<<<<<<<<<<<<<<<<<
    Object[] inputArray = {imgDataYolo};
    Map<Integer, Object> outputMap = new HashMap<>();
    outputMap.put(0, outputRaw0);
    outputMap.put(1, outputRaw1);
    float[][][][][] outputRaw = new float[2][][][][];
    outputRaw[0] = outputRaw0;
    outputRaw[1] = outputRaw1;

    Trace.endSection();

    // Run the inference call.
    Trace.beginSection("run");
    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>YOLOv3-tiny<<<<<<<<<<<<<<<<<<<<<<<<<<
    tfLiteYolo.runForMultipleInputsOutputs(inputArray, outputMap);
    Trace.endSection();


    long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
    Log.d("============== Prediction time Yolov3", ": " + lastProcessingTimeMs);

    List<BoxInfo> lstBoxInfos = new ArrayList<>();

    for(int layer = 0; layer < outputMap.size(); layer++){
      float delta_x = inputSize/GRID_SIZES[layer];
      float delta_y = inputSize/GRID_SIZES[layer];
      float tx = 0.0f, ty = 0.0f, tw = 0.0f, th = 0.0f, bw = 0.0f, bh = 0.0f, bx = 0.0f, by = 0.0f;
      int sizeOfAbox = NUM_CLASSES+5;
      for (int grid_y = 0; grid_y < GRID_SIZES[layer]; grid_y++ ){
        for (int grid_x = 0; grid_x < GRID_SIZES[layer]; grid_x++ ){
          for(int j = 0; j < 3; j++){
             tx = outputRaw[layer][0][grid_y][grid_x][0+j*sizeOfAbox];
             ty = outputRaw[layer][0][grid_y][grid_x][1+j*sizeOfAbox];
             tw = outputRaw[layer][0][grid_y][grid_x][2+j*sizeOfAbox];
             th = outputRaw[layer][0][grid_y][grid_x][3+j*sizeOfAbox];
             bw = ANCHOR_BOXES[2 * j + 0] * (float)Math.exp(tw) / inputSize;
             bh = ANCHOR_BOXES[2 * j + 1] * (float)Math.exp(th) / inputSize;
             bx = (sigmoid(tx)+grid_x)/GRID_SIZES[layer] - bw/2.0f;
             by = (sigmoid(ty)+grid_y)/GRID_SIZES[layer] - bh/2.0f;
             MAX_SCORE score = findScoreYolo(outputRaw[layer][0][grid_y][grid_x], j);
             if(score.score >= scoreThreshold){
                float x_min = bx*inputSize;
                float y_min = by*inputSize;
                float x_max = (bx+bw)*inputSize;
                float y_max = (by+bh)*inputSize;
                if(x_min > 0 && y_min>0 && x_max < inputSize && y_max < inputSize){
                  BoxInfo boxInfo = new BoxInfo(x_min,
                                                y_min,
                                                x_max,
                                                y_max,
                                                score.score,
                                                labels.get(score.indx));
                  lstBoxInfos.add(boxInfo);
                }
             }
          }
        }
      }
    }

    Log.d("====================Detect: ", "value="+lstBoxInfos.size());

    // Show the best detections.
    // after scaling them back to the input size.
    final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
    for(int idx = 0; idx < lstBoxInfos.size(); idx++){
      BoxInfo boxInfo = lstBoxInfos.get(idx);
      final RectF detection = new RectF(boxInfo._xmin, boxInfo._ymin,boxInfo._xmax, boxInfo._ymax);
      recognitions.add(
         new Recognition(
             "" + idx,
             boxInfo._name,
             boxInfo._score,
             detection,
             lastProcessingTimeMs));
    }
    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  @Override
  public void enableStatLogging(final boolean logStats) {}

  @Override
  public String getStatString() {
    return "";
  }

  @Override
  public void close() {
    /** Closes the interpreter and model to release resources. */
    if (tfLiteYolo != null) {
      tfLiteYolo.close();
      tfLiteYolo = null;
    }
    if (gpuDelegateYolo != null) {
      gpuDelegateYolo.close();
      gpuDelegateYolo = null;
    }
  }

  public void setNumThreads(int num_threads) {
    if (tfLiteYolo != null) tfLiteYolo.setNumThreads(num_threads);
  }

  @Override
  public void setUseNNAPI(boolean isChecked) {
    if (tfLiteYolo != null) tfLiteYolo.setUseNNAPI(isChecked);
  }
}
