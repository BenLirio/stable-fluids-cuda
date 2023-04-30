#include <stdio.h>
#include <gif_lib.h>
#include <util/macros.h>
#include <util/state.h>



void gif_write_pixels(GifByteType *dst, float *raw) {
  float *host_raw;
  if (USE_GOLD) {
    host_raw = raw;
  } else {
    host_raw = (float*)malloc(N*sizeof(float));
    cudaMemcpy(host_raw, raw, N*sizeof(float), cudaMemcpyDeviceToHost);
  }
  float *scaled = (float*)malloc(OUTPUT_N*sizeof(float));
  int width_stride = WIDTH / OUTPUT_WIDTH;
  int height_stride = HEIGHT / OUTPUT_WIDTH;
  for (int dst_y = 0; dst_y < OUTPUT_WIDTH; dst_y++) {
    for (int dst_x = 0; dst_x < OUTPUT_WIDTH; dst_x++) {

      int dst_idx = dst_y * OUTPUT_WIDTH + dst_x;
      int src_x = dst_x * width_stride;
      int src_y = dst_y * height_stride;
      scaled[dst_idx] = 0.0f;
      for (int y_off = 0; y_off < height_stride; y_off++) {
        for (int x_off = 0; x_off < width_stride; x_off++) {
          scaled[dst_idx] += host_raw[(src_y + y_off) * WIDTH + (src_x + x_off)];
        }
      }
    }
  }
  float max_val = scaled[0];
  float min_val = scaled[0];
  for (int i = 0; i < OUTPUT_N; i++) {
    max_val = max_val > scaled[i] ? max_val : scaled[i];
    min_val = min_val < scaled[i] ? min_val : scaled[i];
  }
  for (int i = 0; i < OUTPUT_N; i++) {
    float val = (scaled[i]-min_val)/(max_val-min_val);
    GifByteType shade = val * (float)(NUM_SHADES-1);
    dst[i] = shade;
  }
  free(scaled);
}

void gif_write_frame(state_t *state) {
  if (!(OUTPUT&OUTPUT_GIF)) return;
  GifByteType pix[OUTPUT_N];
  gif_write_pixels(pix, state->all_colors[0]->cur);

  SavedImage gif_image;
  gif_image.ImageDesc.Left = 0;
  gif_image.ImageDesc.Top = 0;
  gif_image.ImageDesc.Width = OUTPUT_WIDTH;
  gif_image.ImageDesc.Height = OUTPUT_HEIGHT;
  gif_image.ImageDesc.Interlace = false;
  gif_image.ImageDesc.ColorMap = nullptr;
  gif_image.RasterBits = (GifByteType*)malloc(OUTPUT_N);
  gif_image.ExtensionBlockCount = 0;
  gif_image.ExtensionBlocks = nullptr;
  memcpy(gif_image.RasterBits, pix, OUTPUT_N);

  GifMakeSavedImage(state->gif_dst, &gif_image);
}