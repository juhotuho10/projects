#ifndef NUM_SATELLITES
#define NUM_SATELLITES 0 
#endif
#ifndef WINDOW_WIDTH
#define WINDOW_WIDTH 0 
#endif
#ifndef WINDOW_HEIGHT
#define WINDOW_HEIGHT 0 
#endif
#ifndef SATELLITE_RADIUS
#define SATELLITE_RADIUS 0.0f
#endif
#ifndef BLACK_HOLE_RADIUS
#define BLACK_HOLE_RADIUS 0.0f
#endif


typedef struct { uchar blue; uchar green; uchar red; uchar reserved; } color_u8;

typedef struct {
    float x;
    float y;
} floatvector;

typedef struct {
    float b;
    float g;
    float r;
} color_f32;


inline void ColorBackground(
        __private const float* weights,
        __global const color_f32* sat_identifiers,
        __private color_f32* renderColor,
        float weightTotal)
{
    float red = 0.0f;
    float green = 0.0f;
    float blue = 0.0f;

    for (int k = 0; k < NUM_SATELLITES; ++k)
    {
        float weight_normalized = (weights[k] / weightTotal) * 3.0f;

        red += sat_identifiers[k].r * weight_normalized;
        green += sat_identifiers[k].g * weight_normalized;
        blue += sat_identifiers[k].b * weight_normalized;
    }

    renderColor->r += red;
    renderColor->g += green;
    renderColor->b += blue;
}


inline void GetBackgroundColor(
        float pixel_x,
        float pixel_y,
        __global const floatvector* sat_positions,
        __global const color_f32* sat_identifiers,
        __private float* distances,
        __private float* weights,
        __private float* weightTotal,
        __private color_f32* renderColor)
{
    float minDistance = INFINITY;
    float total = 0.0f;

    // cache weights and distances 
    for (int j = 0; j < NUM_SATELLITES; ++j)
    {
        float xDiff = pixel_x - sat_positions[j].x;
        float yDiff = pixel_y - sat_positions[j].y;

        float dist2 = xDiff*xDiff + yDiff*yDiff;
        float dist  = sqrt(dist2);

        distances[j] = dist;

        float weight = 1.0f / (dist2 * dist2);
        weights[j] = weight;
        total += weight;
    }

    *weightTotal = total;

    // find the closest satellite
    for (int j = 0; j < NUM_SATELLITES; ++j)
    {
        float d = distances[j];
        if (d < minDistance)
        {
            minDistance = d;
            *renderColor = sat_identifiers[j];
        }
    }
}


__kernel void parallel(int mouseX, int mouseY,
        __global const floatvector* sat_positions,
        __global const color_f32* sat_identifiers,
        __global color_u8* pixels,
        int width, int height){                                                                            
   // pixel id         
   int pixel_x = get_global_id(0);
   int pixel_y = get_global_id(1);
   if (pixel_x >= width || pixel_y >= height){
        return;
   }
        

   int idx = pixel_y * WINDOW_WIDTH + pixel_x;

                                    
   private float weights[NUM_SATELLITES];
   private float distances[NUM_SATELLITES];

   color_f32 renderColor;
   renderColor.r = 0.0f;
   renderColor.g = 0.0f;
   renderColor.b = 0.0f;

   float weightTotal = 0.0f;

   GetBackgroundColor(
        (float)pixel_x,
        (float)pixel_y,
        sat_positions,
        sat_identifiers,
        distances,
        weights,
        &weightTotal,
        &renderColor
    );

   int hitsSatellite = 0;
   for (int k = 0; k < NUM_SATELLITES; ++k){
        hitsSatellite |= (distances[k] < SATELLITE_RADIUS);
   }

   color_u8 out;
   out.reserved = 0;

   if (hitsSatellite){
        out.red = 255;
        out.green = 255;
        out.blue = 255;
    } else{ 
        ColorBackground(
            weights,
            sat_identifiers,
            &renderColor,
            weightTotal
        );

        out.red = (uchar)(renderColor.r * 255.0f);
        out.green = (uchar)(renderColor.g * 255.0f);
        out.blue = (uchar)(renderColor.b * 255.0f);
    
    }


    float bh_dx = (float)pixel_x - (float)mouseX;
    float bh_dy = (float)pixel_y - (float)mouseY;
    float distBH = sqrt(bh_dx*bh_dx + bh_dy*bh_dy);

    float mask = (distBH >= BLACK_HOLE_RADIUS) ? 1.0f : 0.0f;

    out.red = (uchar)(out.red * mask);
    out.green = (uchar)(out.green * mask);
    out.blue = (uchar)(out.blue * mask);
        

    pixels[idx] = out;
                  
}    