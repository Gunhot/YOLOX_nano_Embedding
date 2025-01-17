package com.example.tflite_yolov5_test.camera;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.example.tflite_yolov5_test.R;
import com.example.tflite_yolov5_test.camera.env.BorderedText;
import com.example.tflite_yolov5_test.camera.env.ImageUtils;
import com.example.tflite_yolov5_test.camera.tracker.MultiBoxTracker;
import com.example.tflite_yolov5_test.customview.OverlayView;
import com.example.tflite_yolov5_test.TfliteRunner;
import com.example.tflite_yolov5_test.TfliteRunMode;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DetectorActivity extends CameraActivity implements ImageReader.OnImageAvailableListener {


    private static final int TF_OD_API_INPUT_SIZE = 320;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";
    private static final TfliteRunMode.Mode MODE = TfliteRunMode.Mode.NONE_INT8;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
    private static final boolean MAINTAIN_ASPECT = false;
    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private Integer sensorOrientation;

    private TfliteRunner detector;

    private long lastProcessingTimeMs = 0;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private BorderedText borderedText;

    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    protected int getLayoutId() {
        return R.layout.tfe_od_camera_connection_fragment_tracking;
    }


    public float getConfThreshFromGUI() {
        return ((float) ((SeekBar) findViewById(R.id.conf_seekBar2)).getProgress()) / 100.0f;
    }

    public float getIoUThreshFromGUI() {
        return ((float) ((SeekBar) findViewById(R.id.iou_seekBar2)).getProgress()) / 100.0f;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        SeekBar conf_seekBar = (SeekBar) findViewById(R.id.conf_seekBar2);
        Log.d("DetectorActivity", "onCreate: conf_seekBar initialized");

        conf_seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                TextView conf_textView = (TextView) findViewById(R.id.conf_TextView2);
                float thresh = (float) progress / 100.0f;
                conf_textView.setText(String.format("Confidence Threshold: %.2f", thresh));
                Log.d("DetectorActivity", "onProgressChanged: Confidence Threshold set to " + thresh);
                if (detector != null) detector.setConfThresh(thresh);
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                Log.d("DetectorActivity", "onStopTrackingTouch: SeekBar stopped");
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                Log.d("DetectorActivity", "onStartTrackingTouch: SeekBar touched");
            }
        });
        conf_seekBar.setMax(100);
        conf_seekBar.setProgress(30); // 0.25
        SeekBar iou_seekBar = (SeekBar) findViewById(R.id.iou_seekBar2);
        iou_seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                TextView iou_textView = (TextView) findViewById(R.id.iou_TextView2);
                float thresh = (float) progress / 100.0f;
                iou_textView.setText(String.format("IoU Threshold: %.2f", thresh));
                Log.d("DetectorActivity", "onProgressChanged: IoU Threshold set to " + thresh);
                if (detector != null) detector.setIoUThresh(thresh);
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                Log.d("DetectorActivity", "onStopTrackingTouch: SeekBar stopped");
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                Log.d("DetectorActivity", "onStartTrackingTouch: SeekBar touched");
            }
        });
        iou_seekBar.setMax(100);
        iou_seekBar.setProgress(45); // 0.45
    }


    @Override
    protected void setUseNNAPI(final boolean isChecked) {

    }

    @Override
    protected void setNumThreads(final int numThreads) {
        //runInBackground(() -> detector.setNumThreads(numThreads));
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        Log.d("DetectorActivity", "onPreviewSizeChosen: size=" + size + ", rotation=" + rotation);
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector = new TfliteRunner(this, MODE, TF_OD_API_INPUT_SIZE, 0.30f, 0.45f);
            cropSize = TF_OD_API_INPUT_SIZE;
            Log.d("DetectorActivity", "onPreviewSizeChosen: TfliteRunner initialized");
        } catch (final Exception e) {
            e.printStackTrace();
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Detector could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            Log.e("DetectorActivity", "onPreviewSizeChosen: Detector initialization failed", e);
            finish();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();
        sensorOrientation = rotation - getScreenOrientation();
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                    }
                });

        tracker.setFrameConfiguration(getDesiredPreviewFrameSize(), TF_OD_API_INPUT_SIZE, sensorOrientation);
        Log.d("DetectorActivity", "onPreviewSizeChosen: Configuration set");
    }

    @Override
    protected void processImage() {
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        Log.d("DetectorActivity", "processImage: Computing detection started");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        final long nowTime = SystemClock.uptimeMillis();
                        float fps = (float) 1000 / (float) (nowTime - lastProcessingTimeMs);
                        lastProcessingTimeMs = nowTime;
                        Log.d("DetectorActivity", "processImage: FPS calculated - " + fps);

                        detector.setInput(croppedBitmap);
                        final List<TfliteRunner.Recognition> results = detector.runInference();
                        Log.d("DetectorActivity", "processImage: Inference completed with " + results.size() + " results");

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        for (final TfliteRunner.Recognition result : results) {
                            final RectF location = result.getLocation();
                            Log.d("DetectorActivity", "processImage: Detected object at " + location);
                        }

                        tracker.trackResults(results);
                        trackingOverlay.postInvalidate();

                        computingDetection = false;

                        runOnUiThread(
                                new Runnable() {
                                    @Override
                                    public void run() {
                                        TextView fpsTextView = (TextView) findViewById(R.id.textViewFPS);
                                        String fpsText = String.format("FPS: %.2f", fps);
                                        fpsTextView.setText(fpsText);
                                        Log.d("DetectorActivity", "processImage: FPS text updated");

                                        TextView latencyTextView = (TextView) findViewById(R.id.textViewLatency);
                                        latencyTextView.setText(detector.getLastElapsedTimeLog());
                                        Log.d("DetectorActivity", "processImage: Latency text updated");
                                    }
                                });
                    }
                });
    }
}
