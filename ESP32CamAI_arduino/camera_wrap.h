#pragma once

#include "esp_camera.h"
#include "Arduino.h"
#include <string.h>

extern int initCamera();
extern esp_err_t grabImage( size_t& jpg_buf_len, uint8_t *jpg_buf);