package com.p4f.esp32camai.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import com.p4f.esp32camai.MyConstants;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.LinkedList;
import java.util.Vector;

public class TFLiteSegmentationAPIModel extends Classifier{

    public class Segmentation {
        /** Constructor */
        public Segmentation(Bitmap bmp, long timeEx){
            this.bitmap = bmp;
            this.timeExecution = timeEx;
        }

        void copyData(Segmentation seg){
            this.bitmap = seg.getBitmap();
            this.timeExecution = seg.getTimeExecution();
        }

        public Bitmap getBitmap() {
            return bitmap;
        }

        public void setBitmap(Bitmap bitmap) {
            this.bitmap = bitmap;
        }

        public long getTimeExecution(){
            return timeExecution;
        }

        /** Bitmap return from segmentation */
        private Bitmap bitmap;
        /** Time execution of segmentation network inference by tfLite */
        private long timeExecution;
    }

    public Vector<String> getLabels(){
        return labels;
    }

    /** Optional GPU delegate for accleration. */
    private GpuDelegate gpuDelegateSeg = null;

    // Float model
    private static final float IMAGE_STD = 255.0f;
    // Number of threads in the java app
    private static final int NUM_THREADS = 4;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptionsSeg = new Interpreter.Options();

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>for Segmentation<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    private int[] intValuesSeg;
    private int[] inputSizeSeg;
    private ByteBuffer imgDataSeg;
    private float[][][][] outputSeg;
    private byte[][][][] outputSegByteArr;
    private byte[] outputSegProcessed;

    private Interpreter tfLiteSeg;

    final float SEG_THRESH = 0.0f;

    MyConstants.MODEL_TYPE modelTypeSeg = MyConstants.MODEL_TYPE.FLOAT32;

    private Segmentation resultSegmentation;

    private TFLiteSegmentationAPIModel() {}

    public static float sigmoid(float x) {
        return (float)(1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    /** Memory-map the model file in Assets. */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilenameYolo)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilenameYolo);
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

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @param modelFilenameSeg The filepath of the model GraphDef protocol buffer.
     * @param inputSizeSeg The size of image input for Segmentation
     */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilenameSeg,
            final int[] inputSizeSeg,
            final TFLiteSegmentationAPIModel.Device deviceSeg,
            final MyConstants.MODEL_TYPE modelTypeSeg,
            final int queueSize,
            final int height,
            final int width
            )
        throws IOException {
        final TFLiteSegmentationAPIModel d = new TFLiteSegmentationAPIModel();

        d.modelTypeSeg = modelTypeSeg;

        final String strModelSeg  = modelFilenameSeg;

        switch (deviceSeg) {
            case NNAPI:
                d.tfliteOptionsSeg.setUseNNAPI(true);
                break;
            case GPU:
                d.gpuDelegateSeg = new GpuDelegate();
                d.tfliteOptionsSeg.addDelegate(d.gpuDelegateSeg);
                Log.d("GPU delegated for Segmentation","");
                break;
            case CPU:
                break;
        }

        d.tfliteOptionsSeg.setNumThreads(NUM_THREADS);

        try {
            d.tfLiteSeg = new Interpreter(loadModelFile(assetManager, strModelSeg), d.tfliteOptionsSeg);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Pre-allocate buffers.
        int numBytesPerChannelSeg  = 4; // Floating point

        d.inputSizeSeg = inputSizeSeg;
        d.intValuesSeg = new int[d.inputSizeSeg[0] * d.inputSizeSeg[1]];
        d.imgDataSeg = ByteBuffer.allocateDirect(1* d.inputSizeSeg[0]* d.inputSizeSeg[1] * 3 * numBytesPerChannelSeg);
        d.imgDataSeg.order(ByteOrder.nativeOrder());
        d.outputSegProcessed = new byte[d.inputSizeSeg[0] * d.inputSizeSeg[1] * 4];

        d.outputSeg = new float[1][d.inputSizeSeg[1]][d.inputSizeSeg[0]][2];
        d.outputSegByteArr = new byte[1][d.inputSizeSeg[1]][d.inputSizeSeg[0]][2];

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
    public void getResult(Segmentation result){
        synchronized (lockResult){

        }
    }

    @Override
    protected void processing(Bitmap bmp){
        resultSegmentation = recognizeImageSeg(bmp);
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

    public Segmentation recognizeImageSeg(Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.

        Bitmap resizeBitmap = Bitmap.createScaledBitmap(bitmap, inputSizeSeg[0], inputSizeSeg[1], false);

        resizeBitmap.getPixels(intValuesSeg, 0, resizeBitmap.getWidth(), 0, 0, resizeBitmap.getWidth(), resizeBitmap.getHeight());

        imgDataSeg.rewind();
        for (int i = 0; i < inputSizeSeg[1]; ++i) {
            for (int j = 0; j < inputSizeSeg[0]; ++j) {
                int pixelValue = intValuesSeg[i * inputSizeSeg[0] + j];
                if (modelTypeSeg == MyConstants.MODEL_TYPE.UINT8) {
                    // Quantized model
                    imgDataSeg.put((byte) (pixelValue & 0xFF));
                    imgDataSeg.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgDataSeg.put((byte) ((pixelValue >> 16) & 0xFF));
                } else if (modelTypeSeg == MyConstants.MODEL_TYPE.FLOAT32) { // Float model
                    imgDataSeg.putFloat((float)(pixelValue & 0xFF) / IMAGE_STD);
                    imgDataSeg.putFloat((float)((pixelValue >> 8) & 0xFF) / IMAGE_STD);
                    imgDataSeg.putFloat((float)((pixelValue >> 16) & 0xFF)/ IMAGE_STD);
                }
            }
        }

        Trace.endSection(); // preprocessBitmap

        // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Inference<<<<<<<<<<<<<<<<<<<<<<<<<<
        final long startTime = SystemClock.uptimeMillis();
        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        Object[] inputArray = {imgDataSeg};
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        if(modelTypeSeg == MyConstants.MODEL_TYPE.UINT8){
            tfLiteSeg.run(imgDataSeg, outputSegByteArr);
        }else{
            tfLiteSeg.run(imgDataSeg, outputSeg);
        }
        Trace.endSection();

        long lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
        Log.d("============== Prediction time Segmentation", ": " + lastProcessingTimeMs);

//    Log.d("pixel x=105, y=126", ": " + outputSeg[0][126][105][0]);

        for (int i = 0; i < inputSizeSeg[1]; ++i) {
            for (int j = 0; j < inputSizeSeg[0]; ++j) {
                if(modelTypeSeg == MyConstants.MODEL_TYPE.UINT8) {
                    if (outputSegByteArr[0][i][j][0] > SEG_THRESH) {
//            Log.d("debug i j", ": " + i + " " + j);
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4] = (byte) 0;
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4 + 1] = (byte) 200;   //(byte)(outputSeg[0][i][j][0] * 255.0f);
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4 + 2] = (byte) 0;     //(byte)(outputSeg[0][i][j][0] * 255.0f);
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4 + 3] = (byte)((int)outputSegByteArr[0][i][j][0]*50);   //(byte)(outputSeg[0][i][j][0] * 255.0f);
                    } else {
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4 + 3] = 0;
                    }
                }else{
                    if (outputSeg[0][i][j][0] > SEG_THRESH * 1 / 255.0f) {
//            Log.d("debug i j", ": " + i + " " + j);
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4] = (byte) 0;
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4 + 1] = (byte) 200;   //(byte)(outputSeg[0][i][j][0] * 255.0f);
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4 + 2] = (byte) 0;     //(byte)(outputSeg[0][i][j][0] * 255.0f);
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4 + 3] = (byte) (outputSeg[0][i][j][0] * 200.0f);   //(byte)(outputSeg[0][i][j][0] * 255.0f);
                    } else {
                        outputSegProcessed[i * inputSizeSeg[0] * 4 + j * 4 + 3] = 0;
                    }
                }
            }
        }

        Bitmap bmp = Bitmap.createBitmap(inputSizeSeg[0], inputSizeSeg[1], Bitmap.Config.ARGB_8888);
        ByteBuffer buffer = ByteBuffer.wrap(outputSegProcessed);

        bmp.copyPixelsFromBuffer(buffer);

        Segmentation result = new Segmentation(bmp, lastProcessingTimeMs);

        return result;
    }

    @Override
    public void enableStatLogging(final boolean logStats) {}

    @Override
    public String getStatString() {
        return "";
    }

    @Override
    public void close() {
        if (tfLiteSeg != null) {
            tfLiteSeg.close();
            tfLiteSeg = null;
        }
        if (gpuDelegateSeg != null) {
            gpuDelegateSeg.close();
            gpuDelegateSeg = null;
        }
    }

    public void setNumThreads(int num_threads) {
        if (tfLiteSeg!= null) tfLiteSeg.setNumThreads(num_threads);
    }

    @Override
    public void setUseNNAPI(boolean isChecked) {
        if (tfLiteSeg != null) tfLiteSeg.setUseNNAPI(isChecked);
    }
}

