package com.example.android.camera2basic;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.media.Image;
import android.os.SystemClock;
import android.util.Log;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.tensorflow.lite.Interpreter;

public class Classifier {
    /**
     * Tag for the {@link Log}.
     */
    private static final String TAG = "Classifier";

    /** The loaded TensorFlow Lite model. */
    private MappedByteBuffer tfliteModel;

    /** An instance of the driver class to run model inference with Tensorflow Lite. */
    private Interpreter tflite;

    /** Options for configuring the Interpreter. */
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();

    /** Labels corresponding to the output of the vision model. */
    private List<String> labelList;

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private ByteBuffer imgData = null;

    /** Dimensions of inputs. */
    private static final int DIM_BATCH_SIZE = 1;

    private static final int DIM_PIXEL_SIZE = 3;

    private static final String MODEL_PATH = "mobilenet_quant_v1_224.tflite";
    private static final String LABEL_PATH = "labels.txt";
    private static final int ImageSizeX = 224;
    private static final int ImageSizeY = 224;
    private static final int NumBytesPerChannel = 4;
    /** multi-stage low pass filter * */
    private float[] LabelProbArray = null;

    Classifier(Activity activity) throws IOException {
        tfliteModel = loadModelFile(activity);
        tflite = new Interpreter(tfliteModel, tfliteOptions);
        labelList = loadLabelList(activity);
        imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE
                                * getImageSizeX()
                                * getImageSizeY()
                                * DIM_PIXEL_SIZE
                                * getNumBytesPerChannel());
        imgData.order(ByteOrder.nativeOrder());
        LabelProbArray = new float[getNumLabels()];
        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
    }
    private int getNumLabels(){
        return labelList.size();
    }
    private String getModelPath(){
        return MODEL_PATH;
    }
    private String getLabelPath(){
        return LABEL_PATH;
    }
    private int getImageSizeX(){
        return ImageSizeX;
    }
    private int getImageSizeY(){
        return ImageSizeY;
    }
    private int getNumBytesPerChannel(){
        return NumBytesPerChannel;
    }



    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(getModelPath());
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /** Reads label list from Assets. */
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(getLabelPath())));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    /** Closes tflite to release resources. */
    public void close() {
        if(tflite != null)
            tflite.close();
        tflite = null;
        tfliteModel = null;
    }

    /**
     * main routine for classification
     * @param image
     * @return String
     */
    String classify(Image image){

        Log.d(TAG, "Classifier.classify called");
        Random rng = new Random();
        int random_number = rng.nextInt(100 - 0) ;
        return String.valueOf(random_number) + "\n" + String.valueOf(random_number);

        /*
        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.");
            return new String("Uninitialized Classifier.");
        }
        ImageToByteBuffer(image);
        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();
        Inference();
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

        // Print the results.
        String result = printTopK();
        long duration = endTime - startTime;
        result = result + "\n " + duration + "ms";
        return result;
        */
    }
    /*
    private void ImageToByteBuffer(Image image){

    }
    private void Inference(){

    }
    private String printTopK(){

    }
    */
}
