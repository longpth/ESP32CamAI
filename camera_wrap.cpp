#include "camera_wrap.h"

// Select camera model
// #define CAMERA_MODEL_WROVER_KIT
//#define CAMERA_MODEL_ESP_EYE
// #define CAMERA_MODEL_M5STACK_PSRAM
// #define CAMERA_MODEL_M5STACK_WIDE
#define CAMERA_MODEL_AI_THINKER

#include "camera_pins.h"

int initCamera(){
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  //init with high specs to pre-allocate larger buffers
  if(psramFound()){
    config.frame_size = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    // config.frame_size = FRAMESIZE_SVGA;
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return -1;
  }

  sensor_t * s = esp_camera_sensor_get();
  //initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1);//flip it back
    s->set_brightness(s, 1);//up the blightness just a bit
    s->set_saturation(s, -2);//lower the saturation
  }
  //drop down frame size for higher initial frame rate
  // s->set_framesize(s, FRAMESIZE_QVGA);
  // s->set_framesize(s, FRAMESIZE_SVGA);
  s->set_framesize(s, FRAMESIZE_VGA);

#if defined(CAMERA_MODEL_M5STACK_WIDE)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif
  return 0;
}

esp_err_t grabImage( size_t& jpg_buf_len, uint8_t *jpg_buf){
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  fb = esp_camera_fb_get();
  uint8_t *jpg_buf_tmp = NULL;
  if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
  }else{
    if(fb->format != PIXFORMAT_JPEG){
        bool jpeg_converted = frame2jpg(fb, 80, &jpg_buf_tmp, &jpg_buf_len);
        memcpy(jpg_buf, jpg_buf_tmp, jpg_buf_len);
        fb = NULL;
        if(!jpeg_converted){
            Serial.println("JPEG compression failed");
            res = ESP_FAIL;
        }
    } else {
        jpg_buf_len = fb->len;
        memcpy(jpg_buf, fb->buf, jpg_buf_len);
        // Serial.println("Image is in jpg format");
    }
    esp_camera_fb_return(fb);
  }
  return res;
}