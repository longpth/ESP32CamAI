package com.p4f.esp32camai;

import android.app.AlertDialog;
import android.app.Dialog;
import android.content.DialogInterface;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.text.InputType;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Pair;
import android.util.Size;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.fragment.app.Fragment;

import com.p4f.esp32camai.tflite.Classifier;
import com.p4f.esp32camai.tflite.TFLiteObjectDetectionSSDAPIModel;

import java.io.IOException;
import java.io.InputStream;
import java.net.InetAddress;
import java.net.SocketAddress;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import android.graphics.Point;

import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.Tracker;
import org.opencv.tracking.TrackerCSRT;
import org.opencv.tracking.TrackerKCF;
import org.opencv.tracking.TrackerMIL;
import org.opencv.tracking.TrackerMOSSE;
import org.opencv.tracking.TrackerMedianFlow;
import org.opencv.tracking.TrackerTLD;

public class Esp32CameraFragment extends Fragment{

    public enum STATE {
        STOP,
        RUN,
        PAUSE
    };

    enum Drawing{
        DRAWING,
        TRACKING,
        CLEAR,
    }

    final String TAG = "ExCameraFragment";

    private UDPSocket mUdpClient;
    private String mServerAddressBroadCast = "255.255.255.255";
    InetAddress mServerAddr;
    int mServerPort = 6868;
    final byte[] mRequestConnect      = new byte[]{'w','h','o','a','m','i'};
    final byte[] mRequestForward      = new byte[]{'f','o','r','w','a','r','d'};
    final byte[] mRequestForwardTrack = new byte[]{'f','w','t','r','a','c','k'};
    final byte[] mRequestBackward    = new byte[]{'b','a','c','k','w','a','r','d'};
    final byte[] mRequestLeft        = new byte[]{'l','e','f','t'};
    final byte[] mRequestLeftTrack   = new byte[]{'l','e','f','t','t','r','a','c','k'};
    final byte[] mRequestRight       = new byte[]{'r','i','g','h','t'};
    final byte[] mRequestRightTrack  = new byte[]{'r','i','g','h','t','t','r','a','c','k'};
    final byte[] mRequestStop        = new byte[]{'s','t','o','p'};
    final byte[] mRequestCamUp       = new byte[]{'c','a','m','u','p'};
    final byte[] mRequestCamDown     = new byte[]{'c','a','m','d','o','w','n'};
    final byte[] mRequestCamLeft     = new byte[]{'c','a','m','l','e','f','t'};
    final byte[] mRequestCamRight    = new byte[]{'c','a','m','r','i','g','h','t'};
    final byte[] mRequestCamStill    = new byte[]{'c','a','m','s','t','i','l','l'};
    final byte[] mLedOn = new byte[]{'l','e','d','o','n'};
    final byte[] mLedOff = new byte[]{'l','e','d','o','f','f'};
    private Handler handler = new Handler();
    private Bitmap mBitmap;

    ImageView mServerImageView;
    Handler mHandler = new Handler();

    private WebSocketClient mWebSocketClient;
    private String mServerExactAddress;
    private boolean mInitStream = false;
    private boolean mInitTrackObj = false;
    private boolean mStream = false;
    private boolean mObjDet = false;
    private boolean mLed = false;

    ImageButton mBackMoveBtn;
    ImageButton mForMoveBtn;
    ImageButton mRightMoveBtn;
    ImageButton mLeftMoveBtn;

    private Classifier detectorSSD;
    private List<TFLiteObjectDetectionSSDAPIModel.Recognition> detectorSSDResult = new ArrayList<>();

    private final Size CamResolution = new Size(640, 480);

    private OverlayView mTrackingOverlay;
    private Bitmap mBitmapDebug;
    private boolean mProcessing = false;
    private Point[] mPoints = new Point[4];
    private Point mPointCircle = new Point();
    private int mRadiusCircle = 0;
    private Drawing mDrawing = Drawing.CLEAR;
    private boolean mTargetLocked = false;
    private Bitmap mBitmapGrab = null;

    private String mSelectedTracker = "None";
    private String mSelectedTrackerPre = "None";
    private Tracker mTracker;
    private Mat mMatGrabInit;
    private Mat mMatGrab;
    private org.opencv.core.Rect2d mInitRectangle = null;
    private int mBinaryThreshold = 80;
    private int mRadioIndex = 0;
    private Bitmap mBitmapLaneTracking = null;
    //    final String[] mRadioBtnNames = {
//            "None",
//            "TrackerMedianFlow",
//            "TrackerCSRT",
//            "TrackerKCF",
//            "TrackerMOSSE",
//            "TrackerTLD",
//            "TrackerMIL",
//            "LaneTracking"
//    };
    final String[] mRadioBtnNames = {
            "None",
            "ObjectTracking",
            "LaneTracking"
    };

    public void onWindowFocusChanged(){
        int testW = mTrackingOverlay.getWidth();
        int testH = mTrackingOverlay.getHeight();
//        mTrackingOverlay.setLayoutParams(new FrameLayout.LayoutParams(testW, CamResolution.getWidth()/CamResolution.getHeight()*testW));
//        mServerImageView.setLayoutParams(new FrameLayout.LayoutParams(testW, CamResolution.getWidth()/CamResolution.getHeight()*testW));
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        mUdpClient = new UDPSocket(12345);
        mUdpClient.runUdpServer();

        try {
            mServerAddr = InetAddress.getByName(mServerAddressBroadCast);
        }catch (Exception e){

        }

        AssetManager assetManager = getActivity().getAssets();
        if (MyConstants.DEBUG) {
            try {
                InputStream istr = assetManager.open("image1.jpg");
                Bitmap tmpBitmap = BitmapFactory.decodeStream(istr);
                mBitmapDebug = Bitmap.createScaledBitmap(tmpBitmap, CamResolution.getWidth(), CamResolution.getHeight(), false);
            } catch (IOException e) {
                // handle exception
            }
        }

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, getActivity(), null);
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup parent, Bundle savedInstanceState) {
        View rootView = inflater.inflate(R.layout.fragment_camera, parent, false);

        mServerImageView = (ImageView)rootView.findViewById(R.id.imageView);
        Button streamBtn = (Button) rootView.findViewById(R.id.streamBtn);
        streamBtn.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                if (!mStream) {
                    try {
                        mServerAddr = InetAddress.getByName(mServerAddressBroadCast);
                    }catch (Exception e){

                    }
                    mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestConnect);
                    Pair<SocketAddress, String> res = mUdpClient.getResponse();
                    int cnt = 3;
                    while (res.first == null && cnt > 0) {
                        res = mUdpClient.getResponse();
                        cnt--;
                    }
                    if (res.first != null) {
                        Log.d(TAG, res.first.toString() + ":" + res.second);
                        mServerExactAddress = res.first.toString().split(":")[0].replace("/","");
                        mStream = true;
                        connectWebSocket();
                        ((Button) getActivity().findViewById(R.id.streamBtn)).setBackgroundResource(R.drawable.my_button_bg_2);
                        ((Button) getActivity().findViewById(R.id.streamBtn)).setTextColor(Color.rgb(0,0,255));
                        try {
                            mServerAddr = InetAddress.getByName(mServerExactAddress);
                        }catch (Exception e){

                        }
                    }else{
                        Toast toast =
                                Toast.makeText(
                                        getActivity(), "Cannot connect to ESP32 Camera", Toast.LENGTH_LONG);
                        toast.setGravity(Gravity.CENTER, 0, 0);
                        toast.show();
                    }
                } else {
                    mStream = false;
                    mWebSocketClient.close();
                    ((Button) getActivity().findViewById(R.id.streamBtn)).setBackgroundResource(R.drawable.my_button_bg);
                    ((Button) getActivity().findViewById(R.id.streamBtn)).setTextColor(Color.rgb(255,255,255));
                }
            }
        });

        Button ledBtn = (Button) rootView.findViewById(R.id.ledBtn);
        ledBtn.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                if (!mLed) {
                    mLed = true;
                    ((Button) getActivity().findViewById(R.id.ledBtn)).setBackgroundResource(R.drawable.my_button_bg_2);
                    ((Button) getActivity().findViewById(R.id.ledBtn)).setTextColor(Color.rgb(0,0,255));
                    mUdpClient.sendBytes(mServerAddr, mServerPort, mLedOn);
                }else{
                    mLed = false;
                    ((Button) getActivity().findViewById(R.id.ledBtn)).setBackgroundResource(R.drawable.my_button_bg);
                    ((Button) getActivity().findViewById(R.id.ledBtn)).setTextColor(Color.rgb(255,255,255));
                    mUdpClient.sendBytes(mServerAddr, mServerPort, mLedOff);
                }
            }
        });

        Button objFollowBtn = (Button) rootView.findViewById(R.id.objTrackBtn);
        objFollowBtn.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                if(mTargetLocked==true){
                    Toast toast =
                            Toast.makeText(
                                    getActivity(), "Please touch with 2 fingers to release the tracking object first !", Toast.LENGTH_SHORT);
                    toast.show();
                    return;
                }
                trackingDlg();
            }
        });

        Button objDetBtn = (Button) rootView.findViewById(R.id.objDetBtn);
        objDetBtn.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                if (!mObjDet) {
                    ((Button) getActivity().findViewById(R.id.objDetBtn)).setBackgroundResource(R.drawable.my_button_bg_2);
                    ((Button) getActivity().findViewById(R.id.objDetBtn)).setTextColor(Color.rgb(0,0,255));
                } else {
                    ((Button) getActivity().findViewById(R.id.objDetBtn)).setBackgroundResource(R.drawable.my_button_bg);
                    ((Button) getActivity().findViewById(R.id.objDetBtn)).setTextColor(Color.rgb(255,255,255));
                }
                mObjDet = !mObjDet;
            }
        });

        ImageButton.OnTouchListener listener = new ImageButton.OnTouchListener() {
            @Override
            public boolean onTouch(View arg0, MotionEvent event){
                if(event.getAction() == MotionEvent.ACTION_DOWN) {
                    if (((ImageButton) arg0).getId() == R.id.forwardMoveBtn) {
                        ((ImageButton)arg0).setImageResource(R.drawable.ic_btn_forward_on);
                        mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestForward);
                    } else if (((ImageButton) arg0).getId() == R.id.backwardMoveBtn) {
                        ((ImageButton)arg0).setImageResource(R.drawable.ic_btn_backward_on);
                        mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestBackward);
                    } else if (((ImageButton) arg0).getId() == R.id.leftMoveBtn) {
                        ((ImageButton)arg0).setImageResource(R.drawable.ic_btn_left_on);
                        mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestLeft);
                    } else if (((ImageButton) arg0).getId() == R.id.rightMoveBtn) {
                        ((ImageButton)arg0).setImageResource(R.drawable.ic_btn_right_on);
                        mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestRight);
                    }
                    return true;
                }else if(event.getAction() == MotionEvent.ACTION_UP) {
                    if (((ImageButton) arg0).getId() == R.id.forwardMoveBtn) {
                        ((ImageButton)arg0).setImageResource(R.drawable.ic_btn_forward_off);
                    } else if (((ImageButton) arg0).getId() == R.id.backwardMoveBtn) {
                        ((ImageButton)arg0).setImageResource(R.drawable.ic_btn_backward_off);
                    } else if (((ImageButton) arg0).getId() == R.id.leftMoveBtn) {
                        ((ImageButton)arg0).setImageResource(R.drawable.ic_btn_left_off);
                    } else if (((ImageButton) arg0).getId() == R.id.rightMoveBtn) {
                        ((ImageButton)arg0).setImageResource(R.drawable.ic_btn_right_off);
                    }
                    mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestStop);
                    mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestStop);
                    return true;
                }
                return false;
            }
        };

        mBackMoveBtn = (ImageButton)rootView.findViewById(R.id.backwardMoveBtn);
        mBackMoveBtn.setOnTouchListener(listener);
        mForMoveBtn = (ImageButton)rootView.findViewById(R.id.forwardMoveBtn);
        mForMoveBtn.setOnTouchListener(listener);
        mRightMoveBtn = (ImageButton)rootView.findViewById(R.id.rightMoveBtn);
        mRightMoveBtn.setOnTouchListener(listener);
        mLeftMoveBtn = (ImageButton)rootView.findViewById(R.id.leftMoveBtn);
        mLeftMoveBtn.setOnTouchListener(listener);

        try {
            detectorSSD =
                    TFLiteObjectDetectionSSDAPIModel.create(
                            getActivity().getAssets(),
                            "ssdlite_mobilenet_v2_quantized.tflite",
                            "",
                            300,
                            Classifier.Device.CPU,
                            MyConstants.MODEL_TYPE.UINT8,
                            0.5f,
                            1,
                            CamResolution.getWidth(),
                            CamResolution.getHeight()
                    );
            detectorSSD.startThread();
        } catch (final IOException e) {
            Log.e(TAG, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getActivity(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
        }

        for (int i=0; i<mPoints.length;i++){
            mPoints[i] = new Point(0,0);
        }

        mTrackingOverlay = (OverlayView) rootView.findViewById(R.id.tracking_overlay);
        assert (mTrackingOverlay != null);

        mTrackingOverlay.addCallback(
                new OverlayView.DrawCallback() {
                    @Override
                    public void drawCallback(Canvas canvas) {
                        if (MyConstants.DEBUG) {
                            Rect dstRectForRender = new Rect(0, 0, mTrackingOverlay.getWidth(), mTrackingOverlay.getHeight());
                            Matrix matrix = new Matrix();
                            matrix.postRotate(90);
                            Bitmap scaleBitmap = Bitmap.createScaledBitmap(mBitmapDebug, mTrackingOverlay.getWidth(), mTrackingOverlay.getHeight(), false);
                            Bitmap rotatedBitmap = Bitmap.createBitmap(scaleBitmap, 0, 0, mTrackingOverlay.getWidth(), mTrackingOverlay.getHeight(), matrix, true);
                            canvas.drawBitmap(rotatedBitmap, null, dstRectForRender, null);
                        }
                        if (detectorSSD != null && mObjDet) {
                            int overlayWidth = mTrackingOverlay.getWidth();
                            int overlayHeight = mTrackingOverlay.getHeight();
                            int imgWidth = mBitmapGrab.getWidth();
                            int imgHeight = mBitmapGrab.getHeight();
                            ((TFLiteObjectDetectionSSDAPIModel) detectorSSD).getResult(detectorSSDResult);
                            Paint paint = new Paint();
                            Paint paintText = new Paint();
                            paint.setColor(Color.rgb(0, 255, 0));
                            Log.d(TAG, "Obj cnt: " + detectorSSDResult.size());
                            for (TFLiteObjectDetectionSSDAPIModel.Recognition det : detectorSSDResult) {
                                Log.d(TAG, "processing: " + det);
                                paint.setStrokeWidth(10);
                                paint.setStyle(Paint.Style.STROKE);
                                float left = det.getLocation().left * mTrackingOverlay.getWidth();
                                if (left < 0) {
                                    left = 0;
                                } else if (left > mTrackingOverlay.getWidth()) {
                                    left = mTrackingOverlay.getWidth();
                                }

                                float top = det.getLocation().top * mTrackingOverlay.getHeight();
                                if (top < 0) {
                                    top = 0;
                                } else if (top > mTrackingOverlay.getHeight()) {
                                    top = mTrackingOverlay.getHeight();
                                }

                                float right = det.getLocation().right * mTrackingOverlay.getWidth();
                                ;
                                if (right < 0) {
                                    right = 0;
                                } else if (right > mTrackingOverlay.getWidth()) {
                                    right = mTrackingOverlay.getWidth();
                                }

                                float bottom = det.getLocation().bottom * mTrackingOverlay.getHeight();
                                if (bottom < 0) {
                                    bottom = 0;
                                } else if (bottom > mTrackingOverlay.getHeight()) {
                                    bottom = mTrackingOverlay.getHeight();
                                }
                                paintText.setColor(Color.BLUE);
                                paintText.setStrokeWidth(2);
                                paintText.setStyle(Paint.Style.FILL);
                                paintText.setTextSize(50);
                                canvas.drawRect(left, top, right, bottom, paint);
                                paint.setStyle(Paint.Style.FILL);
                                String txt = det.getTitle();// + "(" + String.format("%.2f", det.getConfidence()) + ")";
                                canvas.drawRect(left, top, left-60, top+txt.length()*30+50, paint);
                                canvas.save();
                                canvas.rotate(90, left-50, top + 50);
                                canvas.drawText(txt, left - 50, top + 50, paintText);
                                canvas.restore();
                            }
                        }

                        if(mSelectedTracker.equals("ObjectTracking") && mStream){
                            if(!mInitTrackObj){
                                String msg1 = "Object is selected by 1 touch and drag following by a";
                                String msg2 = "rectangle, make double touch with another finger to lock";
                                String msg3 = "the object, double touch again to release the tracking object";
                                Paint paintText = new Paint();
                                paintText.setColor(Color.YELLOW);
                                paintText.setStrokeWidth(2);
                                paintText.setStyle(Paint.Style.FILL);
                                paintText.setTextSize(mTrackingOverlay.getWidth()/23);
                                canvas.save();
                                canvas.rotate(90, mTrackingOverlay.getWidth()*10/12, mTrackingOverlay.getHeight()/8);
                                canvas.drawText(msg1, mTrackingOverlay.getWidth()*5/6, mTrackingOverlay.getHeight()/8, paintText);
                                canvas.restore();
                                canvas.save();
                                canvas.rotate(90, mTrackingOverlay.getWidth()*9/12, mTrackingOverlay.getHeight()/8);
                                canvas.drawText(msg2, mTrackingOverlay.getWidth()*9/12, mTrackingOverlay.getHeight()/8, paintText);
                                canvas.restore();
                                canvas.save();
                                canvas.rotate(90, mTrackingOverlay.getWidth()*8/12, mTrackingOverlay.getHeight()/8);
                                canvas.drawText(msg3, mTrackingOverlay.getWidth()*8/12, mTrackingOverlay.getHeight()/8, paintText);
                                canvas.restore();
                                canvas.save();
                            }
                            if(mDrawing != Drawing.CLEAR) {
                                Paint paint = new Paint();
                                paint.setColor(Color.rgb(0, 0, 255));
                                paint.setStrokeWidth(10);
                                paint.setStyle(Paint.Style.STROKE);
                                canvas.drawRect(mPoints[0].x, mPoints[0].y, mPoints[1].x, mPoints[1].y, paint);
                                if (mDrawing == Drawing.TRACKING) {
                                    paint.setColor(Color.rgb(0, 255, 0));
                                    canvas.drawLine((mPoints[0].x + mPoints[1].x) / 2,
                                            0,
                                            (mPoints[0].x + mPoints[1].x) / 2,
                                            mTrackingOverlay.getHeight(),
                                            paint);
                                    canvas.drawLine(0,
                                            (mPoints[0].y + mPoints[1].y) / 2,
                                            mTrackingOverlay.getWidth(),
                                            (mPoints[0].y + mPoints[1].y) / 2,
                                            paint);
                                    paint.setColor(Color.YELLOW);
                                    paint.setStrokeWidth(2);
                                    paint.setStyle(Paint.Style.FILL);
                                    paint.setTextSize(30);
                                    String strX = Integer.toString((mPoints[0].x + mPoints[1].x) / 2) + "/" + Integer.toString(mTrackingOverlay.getWidth());
                                    String strY = Integer.toString((mPoints[0].y + mPoints[1].y) / 2) + "/" + Integer.toString(mTrackingOverlay.getHeight());
                                    canvas.drawText(strX, (mPoints[0].x + mPoints[1].x) / 4, (mPoints[0].y + mPoints[1].y) / 2 - 10, paint);
                                    canvas.save();
                                    canvas.rotate(90, (mPoints[0].x + mPoints[1].x) / 2 + 10, (mPoints[0].y + mPoints[1].y) / 4);
                                    canvas.drawText(strY, (mPoints[0].x + mPoints[1].x) / 2 + 10, (mPoints[0].y + mPoints[1].y) / 4, paint);
                                    canvas.restore();
                                }
                            }
                        }else if(mSelectedTracker.equals("LaneTracking") && mStream){
                            Rect dstRectForRender = new Rect(0, 0, mTrackingOverlay.getWidth(), mTrackingOverlay.getHeight());
                            Matrix matrix = new Matrix();
                            matrix.postRotate(90);
                            Bitmap scaleBitmap = Bitmap.createScaledBitmap(mBitmapLaneTracking, mTrackingOverlay.getWidth(), mTrackingOverlay.getHeight(), false);
                            Bitmap rotatedBitmap = Bitmap.createBitmap(scaleBitmap, 0, 0, mTrackingOverlay.getWidth(), mTrackingOverlay.getHeight(), matrix, true);
                            Paint alphaPaint = new Paint();
                            alphaPaint.setAlpha(42);
                            canvas.drawBitmap(rotatedBitmap, null, dstRectForRender, alphaPaint);
                        }else if(mSelectedTracker.equals("ColorTracking") && mStream){
                            Paint paint = new Paint();
                            paint.setColor(Color.argb(50,0, 0, 255));
                            paint.setStrokeWidth(10);
                            paint.setStyle(Paint.Style.FILL);
                            canvas.drawCircle(mPointCircle.x, mPointCircle.y, mRadiusCircle, paint);
                        }else if(mSelectedTracker.equals("None") && mStream){
                            mInitTrackObj = false;
                            //TODO
                            if(!mInitStream){
                                Paint paintText = new Paint();
                                paintText.setColor(Color.YELLOW);
                                paintText.setStrokeWidth(2);
                                paintText.setStyle(Paint.Style.FILL);
                                paintText.setTextSize(mTrackingOverlay.getWidth()/20);
                                canvas.save();
                                canvas.rotate(90, mTrackingOverlay.getWidth()*5/6, mTrackingOverlay.getHeight()/8);
                                canvas.drawText("Touch upper half screen to move camera up !", mTrackingOverlay.getWidth()*5/6, mTrackingOverlay.getHeight()/8, paintText);
                                canvas.restore();
                                canvas.save();
                                canvas.rotate(90, mTrackingOverlay.getWidth()/6, mTrackingOverlay.getHeight()/8);
                                canvas.drawText("Touch lower half screen to move camera down !", mTrackingOverlay.getWidth()/6, mTrackingOverlay.getHeight()/8, paintText);
                                canvas.restore();
                            }
                        }
                    }
                }
        );

        mTrackingOverlay.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent event) {
                final int X = (int) event.getX();
                final int Y = (int) event.getY();
                Log.d(TAG, ": " + Integer.toString(X) + " " + Integer.toString(Y) );
                mInitStream = true;
                mInitTrackObj = true;
                switch (event.getAction() & MotionEvent.ACTION_MASK) {
                    case MotionEvent.ACTION_UP:
//                            Log.d(TAG, ": " +  "MotionEvent.ACTION_UP" );
                        if (mSelectedTracker.equals("None")) {
                            mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestCamStill);
                            break;
                        }
                        if(!mTargetLocked) {
                            mDrawing = Drawing.CLEAR;
                            mTrackingOverlay.postInvalidate();
                        }
                        break;
                    case MotionEvent.ACTION_POINTER_DOWN:
//                            Log.d(TAG, ": " +  "MotionEvent.ACTION_POINTER_DOWN" );
                        if(mSelectedTracker.equals("ObjectTracking")==false){
                            break;
                        }
                        if (mTargetLocked == false) {
                            if((mPoints[0].x-mPoints[1].x != 0) && (mPoints[0].y-mPoints[1].y != 0)) {
                                mTargetLocked = true;
                                mMatGrab = new Mat();
                                Toast toast = Toast.makeText(getActivity(), "Target is LOCKED !", Toast.LENGTH_LONG);
                                toast.setGravity(Gravity.TOP | Gravity.CENTER, 0, 0);
                                toast.show();
                            }else{
                                mTargetLocked = false;
                            }
                        }else{
                            mTargetLocked = false;
                            Toast toast = Toast.makeText(getActivity(), "Target is UNLOCKED !", Toast.LENGTH_LONG);
                            toast.setGravity(Gravity.TOP | Gravity.CENTER, 0, 0);
                            toast.show();
                        }
                        mDrawing = Drawing.DRAWING;
                        mTrackingOverlay.postInvalidate();
                        break;
                    case MotionEvent.ACTION_POINTER_UP:
//                            Log.d(TAG, ": " +  "MotionEvent.ACTION_POINTER_UP" );
                        break;
                    case MotionEvent.ACTION_DOWN:
//                        Log.d(TAG, ": " +  "MotionEvent.ACTION_DOWN" );
                        if (mSelectedTracker.equals("None")) {
                            if (X < mTrackingOverlay.getWidth() / 2) {
                                mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestCamDown);
                            } else {
                                mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestCamUp);
                            }
                            break;
                        }
                        if(!mTargetLocked && mSelectedTracker.equals("ObjectTracking")) {
                            mDrawing = Drawing.DRAWING;
                            mPoints[0].x = X;
                            mPoints[0].y = Y;
                            mPoints[1].x = X;
                            mPoints[1].y = Y;
                            mTrackingOverlay.postInvalidate();
                        }
                        break;
                    case MotionEvent.ACTION_MOVE:
//                            Log.d(TAG, ": " +  "MotionEvent.ACTION_MOVE" );
                        if(!mTargetLocked && mSelectedTracker.equals("ObjectTracking")) {
                            mPoints[1].x = X;
                            mPoints[1].y = Y;
                            mTrackingOverlay.postInvalidate();
                        }
                        break;
                }
//                if(mTargetLocked==true){
//                    getView().findViewById(R.id.objTrackBtn).setEnabled(false);
//                }else{
//                    getView().findViewById(R.id.objTrackBtn).setEnabled(true);
//                }
                return true;
            }
        });

        return rootView;
    }

    private void connectWebSocket() {
        URI uri;
        try {
            uri = new URI("ws://"+mServerExactAddress+":86/");
        } catch (URISyntaxException e) {
            e.printStackTrace();
            return;
        }

        mWebSocketClient = new WebSocketClient(uri) {
            @Override
            public void onOpen(ServerHandshake serverHandshake) {
                Log.d("Websocket", "Open");
            }

            @Override
            public void onClose(int i, String s, boolean b) {
                Log.d("Websocket", "Closed " + s);
            }

            @Override
            public void onMessage(String message){
                Log.d("Websocket", "Receive");
            }

            @Override
            public void onMessage(ByteBuffer message){
//                Log.d("Websocket", "Receive");
                getActivity().runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        byte[] imageBytes= new byte[message.remaining()];
                        message.get(imageBytes);
                        final Bitmap bmp=BitmapFactory.decodeByteArray(imageBytes,0,imageBytes.length);
                        if (bmp == null)
                        {
                            return;
                        }
                        int viewWidth = mServerImageView.getWidth();
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90);
                        final Bitmap bmp_traspose = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true );
                        float imagRatio = (float)bmp_traspose.getHeight()/(float)bmp_traspose.getWidth();
                        int dispViewH = (int)(viewWidth*imagRatio);
                        mServerImageView.setImageBitmap(Bitmap.createScaledBitmap(bmp_traspose, viewWidth, dispViewH, false));

                        mBitmapGrab = bmp;
                        mProcessing = detectorSSD.IsProcessing;
                        if (!mProcessing) {
                            processing();
                        }
                    }
                });
            }

            @Override
            public void onError(Exception e) {
                Log.d("Websocket", "Error " + e.getMessage());
            }
        };
        mWebSocketClient.connect();
    }

    private void trackingDlg(){
        AlertDialog.Builder builder = new AlertDialog.Builder(getActivity());
        builder.setTitle("Tracker Selection");

        final RadioButton[] rb = new RadioButton[mRadioBtnNames.length];
        RadioGroup rg = new RadioGroup(getActivity()); //create the RadioGroup
        rg.setOrientation(RadioGroup.VERTICAL);

        for(int i=0; i < mRadioBtnNames.length; i++){
            rb[i]  = new RadioButton(getActivity());
            rb[i].setText(" " + mRadioBtnNames[i]);
            rb[i].setId(i + 100);
            rg.addView(rb[i]);
            if(mRadioBtnNames[i].equals(mSelectedTracker)){
                rb[i].setChecked(true);
            }
        }

        // This overrides the radiogroup onCheckListener
        rg.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener()
        {
            public void onCheckedChanged(RadioGroup group, int checkedId){
                // This will get the radiobutton that has changed in its check state
                RadioButton checkedRadioButton = (RadioButton)group.findViewById(checkedId);
                // This puts the value (true/false) into the variable
                boolean isChecked = checkedRadioButton.isChecked();
                if (isChecked)
                {
                    // Changes the textview's text to "Checked: example radiobutton text"
                    int i = 0;
                    for( i = 0; i < mRadioBtnNames.length; i++) {
                        if(checkedRadioButton.getText().toString().replace(" ", "").equals(mRadioBtnNames[i])){
                            break;
                        }
                    }
                    mRadioIndex = i;
                }
            }
        });

        LinearLayout lay = new LinearLayout(getActivity());
        lay.setOrientation(LinearLayout.VERTICAL);
        lay.setPadding(0,30,0,0);
        lay.setGravity(Gravity.CENTER_HORIZONTAL);
        lay.addView(rg);

        final TextView labelThresh = new TextView(getActivity());
        labelThresh.setText("Binary Threshold:");
        // Set up the input
        final EditText binThresh = new EditText(getActivity());
        binThresh.setBackground(null);
        // Specify the type of input expected; this, for example, sets the input as a password, and will mask the text
        binThresh.setInputType(InputType.TYPE_CLASS_NUMBER);
        binThresh.setText(Integer.toString(mBinaryThreshold));
        lay.addView(labelThresh);
        lay.addView(binThresh);

        builder.setView(lay);

        // Set up the buttons
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                mSelectedTrackerPre = mSelectedTracker;
                mSelectedTracker = mRadioBtnNames[mRadioIndex];
                if (!mSelectedTracker.equals("None")) {
                    ((Button) getActivity().findViewById(R.id.objTrackBtn)).setBackgroundResource(R.drawable.my_button_bg_2);
                    ((Button) getActivity().findViewById(R.id.objTrackBtn)).setTextColor(Color.rgb(0,0,255));
                } else {
                    ((Button) getActivity().findViewById(R.id.objTrackBtn)).setBackgroundResource(R.drawable.my_button_bg);
                    ((Button) getActivity().findViewById(R.id.objTrackBtn)).setTextColor(Color.rgb(255,255,255));
                }
                mBinaryThreshold = Integer.parseInt(binThresh.getText().toString());
            }
        });


        builder.setCancelable(false);
        Dialog dialog = builder.show();
    }

    private void processing() {

        int overlayWidth = mTrackingOverlay.getWidth();
        int overlayHeight = mTrackingOverlay.getHeight();
        mRadiusCircle = 0;

        if(mObjDet) {
            if (MyConstants.DEBUG) {
                detectorSSD.setBitmap(mBitmapDebug);
            } else {
                detectorSSD.setBitmap(mBitmapGrab);
            }
        }

        if(mSelectedTracker.equals("LaneTracking")){
            if(mMatGrab==null){
                mMatGrab = new Mat();
            }
            Utils.bitmapToMat(mBitmapGrab, mMatGrab);
            org.opencv.imgproc.Imgproc.resize(mMatGrab, mMatGrab, new org.opencv.core.Size(320,240));
            Mat gray = new Mat();
            Mat binary = new Mat();
            org.opencv.imgproc.Imgproc.cvtColor(mMatGrab, gray, Imgproc.COLOR_RGBA2GRAY);
            org.opencv.imgproc.Imgproc.threshold( gray, binary, mBinaryThreshold, 255, 0 );

            int y = binary.rows()*2/3;
            int x0 = -1;
            for(int x = 0; x < binary.cols(); ++x){
                if(binary.get(y,x)[0] < 125){
                    x0 = x;
                    break;
                }
            }

            if(x0 < 0){
                mUdpClient.sendBytes(mServerAddr,mServerPort, mRequestForwardTrack);
            }else if(x0 < binary.width()/2){
                mUdpClient.sendBytes(mServerAddr,mServerPort, mRequestRightTrack);
            }else if(x0 > binary.width()/2) {
                mUdpClient.sendBytes(mServerAddr,mServerPort, mRequestLeftTrack);
            }

            //TODO: DEBUG
            Mat tmp = new Mat();
            try {
                Imgproc.cvtColor(binary, tmp, Imgproc.COLOR_GRAY2BGRA);
                mBitmapLaneTracking = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(tmp, mBitmapLaneTracking);
            }
            catch (CvException e){
                Log.d("Exception",e.getMessage());
            }
        }else if(mSelectedTracker.equals("ColorTracking")){
            if(mMatGrab==null){
                mMatGrab = new Mat();
            }
            Utils.bitmapToMat(mBitmapGrab, mMatGrab);
            org.opencv.imgproc.Imgproc.resize(mMatGrab, mMatGrab, new org.opencv.core.Size(320,240));
            Mat gray = new Mat();
            org.opencv.imgproc.Imgproc.cvtColor(mMatGrab, gray, Imgproc.COLOR_RGBA2GRAY);
            Mat circles = new Mat();
            org.opencv.imgproc.Imgproc.HoughCircles(gray, circles, Imgproc.CV_HOUGH_GRADIENT, 2, gray.rows()/4, 200, 120, 10, 80);
            for (int i = 0; i < circles.cols(); i++) {
                double[] vCircle = circles.get(0, i);

                org.opencv.core.Point pt = new org.opencv.core.Point((int)Math.round(vCircle[0]), (int)Math.round(vCircle[1]));
                int radius = (int)Math.round(vCircle[2]);

                org.opencv.imgproc.Imgproc.circle(gray, pt, radius, new Scalar(255, 0, 0), 2);
                mPointCircle.x = (int)(overlayWidth-pt.y*overlayWidth/gray.rows());
                mPointCircle.y = (int)(pt.x*overlayHeight/gray.cols());
                mRadiusCircle = radius*overlayWidth/gray.rows();
            }
            //TODO: DEBUG
            Bitmap bmp = null;
            Mat tmp = new Mat();
            try {
                Imgproc.cvtColor(gray, tmp, Imgproc.COLOR_GRAY2BGRA);
                bmp = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(tmp, bmp);
            }
            catch (CvException e){
                Log.d("Exception",e.getMessage());
            }
        }else if(mTargetLocked && mSelectedTracker.equals("ObjectTracking")) {
            Utils.bitmapToMat(mBitmapGrab, mMatGrab);
            org.opencv.imgproc.Imgproc.resize(mMatGrab, mMatGrab, new org.opencv.core.Size(320,240));
            org.opencv.imgproc.Imgproc.cvtColor(mMatGrab, mMatGrab, Imgproc.COLOR_RGBA2BGR);

            if(mDrawing==Drawing.DRAWING) {

                int imgWidth = mMatGrab.cols();
                int imgHeight = mMatGrab.rows();

                int x0 = mPoints[0].y;
                int y0 = overlayWidth - mPoints[0].x;
                int x1 = mPoints[1].y;
                int y1 = overlayWidth - mPoints[1].x;

                int minX = (int)((float)Math.min(x0, x1)/overlayHeight*mMatGrab.cols());
                int minY = (int)((float)Math.min(y0, y1)/overlayWidth*mMatGrab.rows());
                int maxX = (int)((float)Math.max(x0, x1)/overlayHeight*mMatGrab.cols());
                int maxY = (int)((float)Math.max(y0, y1)/overlayWidth*mMatGrab.rows());

                mInitRectangle = new org.opencv.core.Rect2d(minX, minY, maxX-minX, maxY-minY);
                mMatGrabInit = new Mat();
                mMatGrab.copyTo(mMatGrabInit);

                if(mSelectedTracker.equals("TrackerMedianFlow")) {
                    mTracker = TrackerMedianFlow.create();
                }else if(mSelectedTracker.equals("TrackerCSRT")||mSelectedTracker.equals("ObjectTracking")) {
                    mTracker = TrackerCSRT.create();
                }else if(mSelectedTracker.equals("TrackerKCF")) {
                    mTracker = TrackerKCF.create();
                }else if(mSelectedTracker.equals("TrackerMOSSE")) {
                    mTracker = TrackerMOSSE.create();
                }else if(mSelectedTracker.equals("TrackerTLD")) {
                    mTracker = TrackerTLD.create();
                }else if(mSelectedTracker.equals("TrackerMIL")) {
                    mTracker = TrackerMIL.create();
                }

                mTracker.init(mMatGrabInit, mInitRectangle);
                mDrawing = Drawing.TRACKING;

                //TODO: DEBUG
//                org.opencv.core.Rect testRect = new org.opencv.core.Rect(minX, minY, maxX-minX, maxY-minY);
//                Mat roi = new Mat(mMatGrab, testRect);
//                Bitmap bmp = null;
//                Mat tmp = new Mat (roi.rows(), roi.cols(), CvType.CV_8U, new Scalar(4));
//                try {
//                    Imgproc.cvtColor(roi, tmp, Imgproc.COLOR_RGB2BGRA);
//                    bmp = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
//                    Utils.matToBitmap(tmp, bmp);
//                }
//                catch (CvException e){
//                    Log.d("Exception",e.getMessage());
//                }

            }else{
                org.opencv.core.Rect2d trackingRectangle = new org.opencv.core.Rect2d(0, 0, 1,1);
                mTracker.update(mMatGrab, trackingRectangle);

//                //TODO: DEBUG
//                org.opencv.core.Rect testRect = new org.opencv.core.Rect((int)trackingRectangle.x,
//                                                                        (int)trackingRectangle.y,
//                                                                        (int)trackingRectangle.width,
//                                                                        (int)trackingRectangle.height);
//                Mat roi = new Mat(mMatGrab, testRect);
//                Bitmap bmp = null;
//                Mat tmp = new Mat (roi.rows(), roi.cols(), CvType.CV_8U, new Scalar(4));
//                try {
//                    Imgproc.cvtColor(roi, tmp, Imgproc.COLOR_RGB2BGRA);
//                    bmp = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
//                    Utils.matToBitmap(tmp, bmp);
//                }
//                catch (CvException e){
//                    Log.d("Exception",e.getMessage());
//                    mTargetLocked = false;
//                    mDrawing = Drawing.DRAWING;
//                }

                mPoints[1].x = overlayWidth - (int)(trackingRectangle.y*(float)overlayWidth/(float)mMatGrab.rows());
                mPoints[0].y = (int)(trackingRectangle.x*(float)mTrackingOverlay.getHeight()/(float)mMatGrab.cols());
                mPoints[0].x = mPoints[1].x - (int)(trackingRectangle.height*(float)mTrackingOverlay.getWidth()/(float)mMatGrab.rows());
                mPoints[1].y = mPoints[0].y +(int)(trackingRectangle.width*(float)mTrackingOverlay.getHeight()/(float)mMatGrab.cols());

                int centerX = (mPoints[0].x+mPoints[1].x)/2;
                int centerY = (mPoints[0].y+mPoints[1].y)/2;
                if(centerX-mTrackingOverlay.getWidth()/2 > 150){
                    mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestCamUp);
                }else if(centerX-mTrackingOverlay.getWidth()/2 < -150){
                    mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestCamDown);
                }else{
                    mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestCamStill);
                }

//                if(centerY-mTrackingOverlay.getHeight()/2 > 200){
//                    Log.d(TAG, ": " + (centerY-mTrackingOverlay.getHeight()/2) );
//                    mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestRightTrack);
//                }else if(centerY-mTrackingOverlay.getHeight()/2 < -200){
//                    Log.d(TAG, ": " + (centerY-mTrackingOverlay.getHeight()/2) );
//                    mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestLeftTrack);
//                }

                mTrackingOverlay.postInvalidate();
            }
        }else{
            if(mSelectedTrackerPre != "None"){
                mUdpClient.sendBytes(mServerAddr, mServerPort, mRequestStop);
            }
            if (mTracker != null) {
                mTracker.clear();
                mTracker = null;
            }
        }
        mSelectedTrackerPre = mSelectedTracker;
        mTrackingOverlay.invalidate();
    }

    public void onDestroy() {
        Log.e(TAG, "onDestroy");
        detectorSSD.requestStop();
        detectorSSD.waitForExit();
        mWebSocketClient.close();
        super.onDestroy();
    }

}

