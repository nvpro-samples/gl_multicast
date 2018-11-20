#pragma once

// general behavior defines

// benchmark mode 
#define BENCHMARK_MODE 0   // default 0
// step width (SAMPLE_LOAD to 20*SAMPLE_LOAD) in benchmark mode 1
#define SAMPLE_LOAD 1  // default 1

// x- and y-tessellation of a torus object
#define TORUS_M 420
#define TORUS_N 420

// bit 0
#define GPUMASK_0 1<<0
// bit 1
#define GPUMASK_1 1<<1

// scene data defines
#define VERTEX_POS        0
#define VERTEX_NORMAL     1

#define UBO_SCENE         0
#define UBO_OBJECT        1

#define TEX_VOL_DATA      0
#define TEX_VOL_DEPTH     1

// compose data defines
#define UBO_COMP          0
#define TEX_COMPOSE_COLOR 0
#define TEX_COMPOSE_DEPTH 8

#ifdef __cplusplus
namespace vertexload
{
#endif

  struct SceneData
  {
    mat4 viewMatrix;      // view matrix: world->view
    mat4 projMatrix;      // proj matrix: view ->proj
    mat4 viewProjMatrix;  // viewproj   : world->proj
    vec3 lightPos_world;  // light position in world space
    vec3 eyepos_world;    // eye position in world space
    vec3 eyePos_view;     // eye position in view space
    vec3 color;           // scene background color

    vec3 objectColor;

    int  loadFactor;

    float projNear;
    float projFar;
  };

  struct ObjectData
  {
    mat4 model;         // model -> world
    mat4 modelView;     // model -> view
    mat4 modelViewIT;   // model -> view for normals
    mat4 modelViewProj; // model -> proj
    vec3 color;         // model color
  };

  struct ComposeData 
  {
    int in_width;   // width of the input textures
    int in_height;  // height of the input textures
    int out_width;  // width of the output buffer
    int out_height; // height of the output buffer
  };

#ifdef __cplusplus
}
#endif

#if defined(GL_core_profile) || defined(GL_compatibility_profile) || defined(GL_es_profile)
// prevent this to be used by c++

#if defined(USE_SCENE_DATA)
layout(std140,binding=UBO_SCENE) uniform sceneBuffer {
  SceneData scene;
};
layout(std140,binding=UBO_OBJECT) uniform objectBuffer {
  ObjectData object;
};
#endif

#if defined(USE_COMPOSE_DATA)
layout(std140,binding=UBO_SCENE) uniform composeBuffer {
  ComposeData compose;
};
#endif

#endif


/*-----------------------------------------------------------------------
  Copyright (c) 2014, NVIDIA. All rights reserved.
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:
   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Neither the name of its contributors may be used to endorse 
     or promote products derived from this software without specific
     prior written permission.
  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------*/

