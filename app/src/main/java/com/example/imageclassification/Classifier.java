package com.example.imageclassification;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;

import androidx.annotation.NonNull;

import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

class Classifier {
    private Interpreter interpreter;
    private List<String> labelList;
    private int INPUT_SIZE ;
    private int PIXEL_SIZE = 3;
    private int IMAGE_MEAN = 0;
    private float IMAGE_STD = 255.0f;
    private float MAX_RESULTS = 3;
    private float THRESHOLD = 0.1f;

    //TODO delegates
    GpuDelegate gpuDelegate;
    HexagonDelegate hexagonDelegate;
    Classifier(AssetManager assetManager, String modelPath, String labelPath, int inputSize, Context context) throws IOException {
        INPUT_SIZE = inputSize;
        Interpreter.Options options = new Interpreter.Options();

        //TODO GPU delegate
//        CompatibilityList compatList = new CompatibilityList();
//        if(compatList.isDelegateSupportedOnThisDevice()){
//            // if the device has a supported GPU, add the GPU delegate
//            GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
//            GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
//            options.addDelegate(gpuDelegate);
//        } else {
//            // if the GPU is not supported, run on 4 threads
//            options.setNumThreads(4);
//        }

        //TODO Hexagon delegate
        // Create the Delegate instance.
//        try {
//            hexagonDelegate = new HexagonDelegate(context);
//            options.addDelegate(hexagonDelegate);
//        } catch (UnsupportedOperationException e) {
//            // Hexagon delegate is not supported on this device.
//            options.setNumThreads(4);
//
//        }

        //TODO NNAPI
//        NnApiDelegate nnApiDelegate = new NnApiDelegate();
//        options.addDelegate(nnApiDelegate);
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        labelList = loadLabelList(assetManager, labelPath);
    }


    class Recognition{
         String id= "";
        String title= "";
        float confidence= 0F;

        public Recognition(String i, String s, float confidence) {
            id= i;
            title = s;
            this.confidence = confidence;
        }

        @NonNull
        @Override
        public String toString() {
            return "Title = "+title+", Confidence = "+confidence;
        }
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

        /**
         * Returns the result after running the recognition with the help of interpreter
         * on the passed bitmap
         */
    List<Recognition> recognizeImage(Bitmap bitmap) {
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
        float [][]result = new float[1][labelList.size()];
        long startTimeForReference = SystemClock.uptimeMillis();
        interpreter.run(byteBuffer, result);
        long endTimeForReference = SystemClock.uptimeMillis();
        Log.d("tryTimeCost","Timecost: "+(endTimeForReference-startTimeForReference));
        return getSortedResultFloat(result);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        byteBuffer = ByteBuffer.allocateDirect(4  * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    public void close(){
        if (hexagonDelegate != null) {
            hexagonDelegate.close();
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
        }

        if (interpreter != null) {
            interpreter.close();
        }

    }

    @SuppressLint("DefaultLocale")
    private List<Recognition> getSortedResultFloat(float[][] labelProbArray) {
        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        (int) MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.confidence, lhs.confidence);
                            }
                        });

        for(int i = 0; i < labelList.size(); ++i) {
            float confidence = labelProbArray[0][i];
            Log.d("tryCon",confidence+"");
            if (confidence > THRESHOLD) {
                pq.add(new Recognition(""+ i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        confidence));
            }
        }
        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = (int) Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        return recognitions;
    }

}
