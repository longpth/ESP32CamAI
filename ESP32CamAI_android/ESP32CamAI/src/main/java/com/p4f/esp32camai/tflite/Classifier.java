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

import android.graphics.Bitmap;

import java.util.Queue;

/** Generic interface for interacting with different recognition engines. */
public class Classifier implements Runnable {

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    NNAPI,
    GPU
  };

  @Override
  public void run(){
    Bitmap.Config conf = Bitmap.Config.ARGB_8888; // see other conf types
    Bitmap bmp = Bitmap.createBitmap(width, height, conf); // this creates a MUTABLE bitmap
    try{

      while(!isStop()){
        if(getBitmap(bmp)){
          processing(bmp);
          IsProcessing = false;
        }
      }

    }catch(Exception e){
      System.out.println(e);
    }
  }

  void enableStatLogging(final boolean debug){

  }

  String getStatString(){
    return "";
  }

  void close(){

  }

  public void setNumThreads(int num_threads){

  }

  public void setUseNNAPI(boolean isChecked){

  }

  /**
   * Add bitmap to queue, to be used for cnn inference
   *
   * @param bmp The bitmap to be add to queue.
   */
  public void setBitmap(Bitmap bmp){
    synchronized (bmpQueue){
      IsProcessing = true;
      bmpQueue.add(bmp);
    }
  }

  /**
   * Get bitmap from queue
   *
   * @param bmp The bitmap to be taken from queue.
   */
  protected boolean getBitmap(Bitmap bmp){
    boolean ret = false;
    synchronized (bmpQueue) {
      if(bmpQueue.size()>0) {
        Bitmap tmp = bmpQueue.poll();
        int[] pixels = new int[tmp.getWidth() * tmp.getHeight()];
        tmp.getPixels(pixels, 0, tmp.getWidth(), 0, 0, tmp.getWidth(), tmp.getHeight());
        bmp.setPixels(pixels, 0, bmp.getWidth(), 0, 0, bmp.getWidth(), bmp.getHeight());
        ret = true;
      }else{
        ret = false;
      }
    }
    return ret;
  }

  /**
   * Request stop worker thread
   */
  public void requestStop(){
    synchronized(isRequestStop){
      isRequestStop = true;
    }
  }

  /**
   * Check if worker was request to be stopped
   */
  public Boolean isStop(){
    boolean ret;
    synchronized(isRequestStop) {
      ret = isRequestStop;
    }
    return ret;
  }

  /**
   * Start object detection thread
   */
  public void startThread(){
    threadDectection.start();
  }

  /**
   * Wait for thread exit
   */
  public void waitForExit(){
    try {
      threadDectection.join();
    } catch (InterruptedException e){

    }
  }

  /**
   * processing the bitmap
   *
   * @param bmp The bitmap to be taken for processing.
   */
  protected void processing(Bitmap bmp){

  }

  /*Bitmaps queue*/
  protected  Queue<Bitmap> bmpQueue;

  protected  Thread threadDectection;

  protected Boolean isRequestStop = false;

  protected final Object lockResult = new Object();

  protected int width, height;

  public boolean IsProcessing = false;

}
