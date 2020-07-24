/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* Contact iesser@nvidia.com (Ingo Esser) for feedback */

#include <include_gl.h>

#include <imgui/imgui_helper.h>
#include <imgui/imgui_impl_gl.h>

#include <nvh/cameracontrol.hpp>
#include <nvh/geometry.hpp>
#include <nvgl/base_gl.hpp>
#include <nvgl/programmanager_gl.hpp>
#include <nvgl/appwindowprofiler_gl.hpp>
#include <nvmath/nvmath_glsltypes.h>

#include <chrono>
#include <iostream>
#include <locale>
#include <thread>

using namespace nvmath;
#include "common.h"

namespace
{
  int const SAMPLE_SIZE_WIDTH(800);
  int const SAMPLE_SIZE_HEIGHT(400);

  // on screen and offscreen sizes are decoupled:
  int const SAMPLE_OFFSCREEN_SIZE_WIDTH(  SAMPLE_SIZE_WIDTH/2 );
  int const SAMPLE_OFFSCREEN_SIZE_HEIGHT( SAMPLE_SIZE_HEIGHT );

  int const SAMPLE_MAJOR_VERSION(4);
  int const SAMPLE_MINOR_VERSION(5);
}

namespace vertexload
{

  namespace render
  {

    struct UIData {
      bool m_drawUI = true;
      int  m_loadFactor = 102;
      bool m_multicast = true;
      bool m_multicastCopy = true;
      bool m_multicastBlit = false;
    };

    struct Vertex {
      Vertex(const nvh::geometry::Vertex& vertex){
        position  = vertex.position;
        normal    = vertex.normal;
        color     = nvmath::vec4(1.0f);
      }

      nvmath::vec4   position;
      nvmath::vec4   normal;
      nvmath::vec4   color;
    };

    struct Buffers
    {
      Buffers()
        : vbo(0)
        , ibo(0)
        , sceneUbo(0)
        , objectUbo(0)
        , composeUbo(0)
        , numVertices(0)
        , numIndices(0)
      {}

      GLuint  vbo;
      GLuint  ibo;
      GLuint  sceneUbo;
      GLuint  objectUbo;
      GLuint  composeUbo;

      GLsizei numVertices;
      GLsizei numIndices;
    };

    struct Textures
    {
      Textures()
        : colorTexLeft(0)
        , colorTexRight(0)
        , depthTex(0)
      {}

      GLuint colorTexLeft;
      GLuint colorTexRight;
      GLuint depthTex;
    };

    struct Programs
    {
      nvgl::ProgramID scene;
      nvgl::ProgramID compose;
    };

    struct Data
    {
      Data()
        : windowWidth( SAMPLE_SIZE_WIDTH )
        , windowHeight( SAMPLE_SIZE_HEIGHT )
        , texWidth( SAMPLE_OFFSCREEN_SIZE_WIDTH )
        , texHeight( SAMPLE_OFFSCREEN_SIZE_HEIGHT )
      {
        sceneData.projNear = 0.01f;
        sceneData.projFar  = 100.0f;
      }

      UIData                    m_uiData;
      UIData                    m_lastUIData;
      ImGuiH::Registry          m_ui;
      double                    m_uiTime = 0;

      Buffers   buf;
      Textures  tex;
      Programs  prog; 

      SceneData   sceneData;
      ObjectData  objectData;
      ComposeData composeData;

      GLuint renderFBO;
      GLuint tempFBO;

      nvgl::ProgramManager pm;

      int numGPUs;
      int windowWidth;
      int windowHeight;
      int texWidth;
      int texHeight;
    };

    auto initPrograms( Data& rd ) -> bool
    {
      nvgl::ProgramManager& pm = rd.pm;
      Programs& programs = rd.prog;

      bool validated(true);
      pm.addDirectory( std::string("GLSL_" PROJECT_NAME) );
      pm.addDirectory( NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY) );

      pm.registerInclude("common.h", "common.h");

      {
        programs.scene = pm.createProgram(
          nvgl::ProgramManager::Definition(GL_VERTEX_SHADER,   "#define USE_SCENE_DATA", "scene.vert.glsl"),
          nvgl::ProgramManager::Definition(GL_FRAGMENT_SHADER, "#define USE_SCENE_DATA", "scene.frag.glsl"));
      }

      {
        programs.compose = pm.createProgram(
          nvgl::ProgramManager::Definition(GL_VERTEX_SHADER,   "#define USE_COMPOSE_DATA", "compose.vert.glsl"),
          nvgl::ProgramManager::Definition(GL_FRAGMENT_SHADER, "#define USE_COMPOSE_DATA", "compose.frag.glsl"));
      }

      validated = pm.areProgramsValid();
      return validated;
    }

    auto initFBOs( Data& rd ) -> void
    {
      nvgl::newFramebuffer( rd.renderFBO );
      nvgl::newFramebuffer( rd.tempFBO );
    }

    auto initBuffers( Data& rd ) -> void
    {
      Buffers& buffers = rd.buf;

      // Torus geometry
      {
        unsigned int m = TORUS_M;
        unsigned int n = TORUS_N;
        float innerRadius = 0.8f;
        float outerRadius = 0.2f;

        std::vector< nvmath::vec3 > vertices;
        std::vector< nvmath::vec3 > tangents;
        std::vector< nvmath::vec3 > binormals;
        std::vector< nvmath::vec3 > normals;
        std::vector< nvmath::vec2 > texcoords;
        std::vector<unsigned int> indices;

        unsigned int size_v = ( m + 1 ) * ( n + 1 );

        vertices.reserve( size_v );
        tangents.reserve( size_v );
        binormals.reserve( size_v );
        normals.reserve( size_v );
        texcoords.reserve( size_v );
        indices.reserve( 6 * m * n );

        float mf = (float) m;
        float nf = (float) n;

        float phi_step   = 2.0f * nv_pi / mf;
        float theta_step = 2.0f * nv_pi / nf;

        // Setup vertices and normals
        // Generate the Torus exactly like the sphere with rings around the origin along the latitudes.
        for ( unsigned int latitude = 0; latitude <= n; latitude++ ) // theta angle
        {
          float theta = (float) latitude * theta_step;
          float sinTheta = sinf(theta);
          float cosTheta = cosf(theta);

          float radius = innerRadius + outerRadius * cosTheta;

          for ( unsigned int longitude = 0; longitude <= m; longitude++ ) // phi angle
          {
            float phi = (float) longitude * phi_step;
            float sinPhi = sinf(phi);
            float cosPhi = cosf(phi);

            vertices.push_back( nvmath::vec3( radius      *  cosPhi, 
              outerRadius *  sinTheta, 
              radius      * -sinPhi ) );

            tangents.push_back( nvmath::vec3( -sinPhi, 0.0f, -cosPhi ) );

            binormals.push_back( nvmath::vec3( cosPhi * -sinTheta,
              cosTheta, 
              sinPhi * sinTheta ) );

            normals.push_back( nvmath::vec3( cosPhi * cosTheta,
              sinTheta,  
              -sinPhi * cosTheta ) );

            texcoords.push_back( nvmath::vec2( (float) longitude / mf , (float) latitude / nf ) );
          }
        }

        const unsigned int columns = m + 1;

        // Setup indices
        for( unsigned int latitude = 0 ; latitude < n ; latitude++ ) 
        {
          for( unsigned int longitude = 0 ; longitude < m ; longitude++ )
          {
            // two triangles
            indices.push_back(  latitude      * columns + longitude     );  // lower left
            indices.push_back(  latitude      * columns + longitude + 1 );  // lower right
            indices.push_back( (latitude + 1) * columns + longitude     );  // upper left

            indices.push_back( (latitude + 1) * columns + longitude     );  // upper left
            indices.push_back(  latitude      * columns + longitude + 1 );  // lower right
            indices.push_back( (latitude + 1) * columns + longitude + 1 );  // upper right
          }
        }

        buffers.numVertices = static_cast<GLsizei>(vertices.size());
        GLsizeiptr const sizePositionAttributeData = vertices.size() * sizeof(vertices[0]);
        GLsizeiptr const sizeNormalAttributeData   = normals.size()  * sizeof(normals[0]);

        buffers.numIndices = static_cast<GLsizei>(indices.size());
        GLsizeiptr sizeIndexData = indices.size() * sizeof(indices[0]);

        nvgl::newBuffer( buffers.vbo );
        glBindBuffer   ( GL_ARRAY_BUFFER, buffers.vbo );
        glBufferData   ( GL_ARRAY_BUFFER, sizePositionAttributeData + sizeNormalAttributeData, nullptr , GL_STATIC_DRAW );
        glBufferSubData( GL_ARRAY_BUFFER, 0                        , sizePositionAttributeData, &vertices[0] );
        glBufferSubData( GL_ARRAY_BUFFER, sizePositionAttributeData, sizeNormalAttributeData,   &normals[0] );
        glBindBuffer   ( GL_ARRAY_BUFFER, 0 );

        nvgl::newBuffer( buffers.ibo );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, buffers.ibo );
        glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeIndexData, &indices[0], GL_STATIC_DRAW );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
      }


      // GL_NV_gpu_multicast
      // need to mark UBOs GL_PER_GPU_STORAGE_BIT_NV to make sure they can contain different data per GPU
      nvgl::newBuffer( buffers.sceneUbo );
      glNamedBufferStorage( buffers.sceneUbo, sizeof(SceneData), nullptr, GL_DYNAMIC_STORAGE_BIT|GL_PER_GPU_STORAGE_BIT_NV );


      nvgl::newBuffer( buffers.objectUbo );
      glBindBuffer( GL_UNIFORM_BUFFER, buffers.objectUbo );
      glBufferData( GL_UNIFORM_BUFFER, sizeof(ObjectData), nullptr, GL_DYNAMIC_DRAW );
      glBindBuffer( GL_UNIFORM_BUFFER, 0 );

      nvgl::newBuffer( buffers.composeUbo );
      glBindBuffer( GL_UNIFORM_BUFFER, buffers.composeUbo );
      glBufferData( GL_UNIFORM_BUFFER, sizeof(ComposeData), nullptr, GL_DYNAMIC_DRAW );
      glBindBuffer( GL_UNIFORM_BUFFER, 0 );
    }

    auto initTextures( Data& rd ) -> void
    {
      auto newTex = [&]( GLuint& tex )
      {
        nvgl::newTexture( tex, GL_TEXTURE_2D );
        glBindTexture ( GL_TEXTURE_2D, tex );
        glTexStorage2D( GL_TEXTURE_2D, 1, GL_RGBA8, rd.texWidth, rd.texHeight );
        glBindTexture ( GL_TEXTURE_2D, 0);


        // GL_NV_gpu_multicast
        // we need to clear the textures via an FBO once to get a P2P flag - this allows texture copies between GPUs
        glBindFramebuffer( GL_FRAMEBUFFER, rd.tempFBO );
        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0 );
        glClearBufferfv( GL_COLOR, 0, &vec4f(0.0f)[0] );
        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0 );
        glBindFramebuffer( GL_FRAMEBUFFER, 0 );


      };

      newTex( rd.tex.colorTexLeft );
      newTex( rd.tex.colorTexRight );

      nvgl::newTexture(rd.tex.depthTex, GL_TEXTURE_2D);
      glBindTexture  ( GL_TEXTURE_2D, rd.tex.depthTex );
      glTexStorage2D ( GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT24, rd.texWidth, rd.texHeight);
      glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
      glBindTexture  ( GL_TEXTURE_2D, 0 );
    }

    auto renderTori( Data& rd, size_t numTori, size_t width, size_t height, mat4f view ) -> void
    {
      float num = (float)numTori;

      // bind geometry
      glBindBuffer( GL_ARRAY_BUFFER, rd.buf.vbo );
      glVertexAttribPointer( VERTEX_POS,    3, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0 );
      glVertexAttribPointer( VERTEX_NORMAL, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (GLvoid*)(rd.buf.numVertices*3*sizeof(float)) );

      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, rd.buf.ibo );

      glEnableVertexAttribArray( VERTEX_POS );
      glEnableVertexAttribArray( VERTEX_NORMAL );

      // distribute num tori into an numX x numY pattern
      // with numX * numY > num, numX = aspect * numY

      float aspect = (float)width / (float)height;

      size_t numX = static_cast<size_t>( ceil(sqrt(num * aspect)) );
      size_t numY = static_cast<size_t>( (float)numX/aspect );
      if( numX * numY < num )
      {
        ++numY;
      }
      //size_t numY = static_cast<size_t>( ceil(sqrt(num / aspect)) );
      float rx = 1.0f;                     // radius of ring
      float ry = 1.0f;
      float dx = 1.0f;                     // ring distance
      float dy = 1.5f;
      float sx = (numX - 1) * dx + 2 * rx; // array size 
      float sy = (numY - 1) * dy + 2 * ry;

      float x0 = -sx / 2.0f + rx;
      float y0 = -sy / 2.0f + ry;

      float scale = std::min( 1.f/sx , 1.f/sy ) * 0.8f;

      size_t torusIndex = 0;
      for( size_t i = 0; i < numY && torusIndex < num; ++i )
      {
        for( size_t j = 0; j < numX && torusIndex < num; ++j )
        {
          float y = y0 + i * dy;
          float x = x0 + j * dx;
          rd.objectData.model = nvmath::scale_mat4( nvmath::vec3(scale) ) * nvmath::translation_mat4( nvmath::vec3( x, y, 0.0f ) ) * nvmath::rotation_mat4_x( (j%2?-1.0f:1.0f) * 45.0f * nv_pi/180.0f );

          rd.objectData.modelView = view * rd.objectData.model;
          rd.objectData.modelViewIT = nvmath::transpose(nvmath::invert(rd.objectData.modelView));
          rd.objectData.modelViewProj = rd.sceneData.viewProjMatrix * rd.objectData.model;

          rd.objectData.color = nvmath::vec3f( (torusIndex+1)&1, ((torusIndex+1)&2)/2, ((torusIndex+1)&4)/4 );

          // set model UBO
          glNamedBufferSubData( rd.buf.objectUbo, 0, sizeof(ObjectData), &rd.objectData );
          glBindBufferBase(GL_UNIFORM_BUFFER, UBO_OBJECT, rd.buf.objectUbo);

          glDrawElements( GL_TRIANGLES, rd.buf.numIndices, GL_UNSIGNED_INT, NV_BUFFER_OFFSET(0) );

          ++torusIndex;
        }
      }

      glBindBuffer( GL_ARRAY_BUFFER, 0 );
      glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

      glDisableVertexAttribArray( VERTEX_POS );
      glDisableVertexAttribArray( VERTEX_NORMAL );
    }
  } //namespace render

  class Sample : public nvgl::AppWindowProfilerGL {
  public:
    Sample();

    bool begin();
    void processUI(double time);
    void think(double time);
    void resize(int width, int height);
    void end();

    // return true to prevent m_windowState updates
    bool mouse_pos(int x, int y) {
      if (!m_rd.m_uiData.m_drawUI) return false;
      return ImGuiH::mouse_pos(x, y);
    }
    bool mouse_button(int button, int action) {
      if (!m_rd.m_uiData.m_drawUI) return false;
      return ImGuiH::mouse_button(button, action);
    }
    bool mouse_wheel(int wheel) {
      if (!m_rd.m_uiData.m_drawUI) return false;
      return ImGuiH::mouse_wheel(wheel);
    }
    bool key_char(int button) {
      if (!m_rd.m_uiData.m_drawUI) return false;
      return ImGuiH::key_char(button);
    }
    bool key_button(int button, int action, int mods) {
      if (!m_rd.m_uiData.m_drawUI) return false;
      return ImGuiH::key_button(button, action, mods);
    }

  protected:    
    nvh::CameraControl m_control;
    size_t                    m_frameCount;
    render::Data              m_rd;
  };

  Sample::Sample()
    : nvgl::AppWindowProfilerGL( /*singleThreaded=*/true, /*doSwap=*/true )
    , m_frameCount( 0 )
  {
    // set the environment variable GL_NV_gpu_multicast to tell the driver to switch to multicast mode
    putenv("GL_NV_GPU_MULTICAST=1");
  }

  bool Sample::begin()
  {
    std::cout << "Setting locale to " << std::locale("").name().c_str() << "\n";
    std::locale::global(std::locale(""));
    std::cout.imbue(std::locale());

    ImGuiH::Init(m_windowState.m_winSize[0], m_windowState.m_winSize[1], this);
    ImGui::InitGL();

#pragma message(__FILE__ "(483): Fix the vsync()")
    //vsync( false );
    
    bool validated( true );


    // GL_NV_gpu_multicast
    if ( !has_GL_NV_gpu_multicast )
    {
      minimize();
      std::cout << "Extension GL_NV_gpu_multicast not found!\n";
      std::this_thread::sleep_for(std::chrono::seconds(5));
      return false;
    }

    // query the number of GPUs available in this system and configured for NV multicast
    glGetIntegerv ( GL_MULTICAST_GPUS_NV, &m_rd.numGPUs );
    std::cout << "GPUs found: " << m_rd.numGPUs << "\n";
    
    // control setup
    m_control.m_sceneOrbit = nvmath::vec3(0.0f);
    m_control.m_sceneDimension = 1.0f;
    m_control.m_viewMatrix = nvmath::look_at(m_control.m_sceneOrbit - vec3( 0, 0 ,-m_control.m_sceneDimension ), m_control.m_sceneOrbit, vec3(0,1,0));

    render::initPrograms( m_rd );
    render::initFBOs( m_rd );
    render::initBuffers( m_rd );
    render::initTextures( m_rd );

    std::cout << "Scene data: \n";
    std::cout << "Vertices per torus:  " << m_rd.buf.numVertices << "\n";
    std::cout << "Triangles per torus: " << m_rd.buf.numIndices/3 << "\n";

    glEnable( GL_DEPTH_TEST );
    glEnable( GL_CULL_FACE );
    glFrontFace( GL_CCW );
    
    return validated;
  }

  void Sample::processUI(double time)
  {
    int width = m_windowState.m_winSize[0];
    int height = m_windowState.m_winSize[1];

    // Update imgui configuration
    auto &imgui_io = ImGui::GetIO();
    imgui_io.DeltaTime = static_cast<float>(time - m_rd.m_uiTime);
    imgui_io.DisplaySize = ImVec2(width, height);

    m_rd.m_uiTime = time;

    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(5, 5), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 0), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("NVIDIA " PROJECT_NAME, nullptr)) {
      ImGui::PushItemWidth(150);

      ImGui::InputInt("loadFactor", &m_rd.m_uiData.m_loadFactor, 1, 10);
      ImGui::Checkbox("multicast", &m_rd.m_uiData.m_multicast);
      if (m_rd.m_uiData.m_multicast)
      {
        ImGui::Checkbox("multicast copy", &m_rd.m_uiData.m_multicastCopy);
        ImGui::Checkbox("multicast blit", &m_rd.m_uiData.m_multicastBlit);
      }
      ImGui::End();
    }
  }

  void Sample::think(double time)
  {
    NV_PROFILE_GL_SECTION("Frame");

    processUI(time);

    // detect ui data changes here

    m_rd.m_lastUIData = m_rd.m_uiData;

    // handle mouse input
    m_control.processActions(m_windowState.m_winSize,
      nvmath::vec2f(m_windowState.m_mouseCurrent[0],m_windowState.m_mouseCurrent[1]),
      m_windowState.m_mouseButtonFlags, m_windowState.m_mouseWheel);

    // handle keyboard inputs, change number of objects
    if( m_windowState.onPress(KEY_UP) )
    {
      m_rd.m_uiData.m_loadFactor += SAMPLE_LOAD;
      std::cout << "loadFactor: " << m_rd.m_uiData.m_loadFactor << std::endl;
    }
    if( m_windowState.onPress(KEY_DOWN) )
    {
      if(m_rd.m_uiData.m_loadFactor > SAMPLE_LOAD )
      {
        m_rd.m_uiData.m_loadFactor -= SAMPLE_LOAD;
      }
      std::cout << "loadFactor: " << m_rd.m_uiData.m_loadFactor << std::endl;
    }
    if( m_windowState.onPress(KEY_SPACE) )
    {
      m_rd.m_uiData.m_drawUI = !m_rd.m_uiData.m_drawUI;
    }
    
    ++m_frameCount;

    // use the windows width and height here to make sure the aspect ratio in the output image is correct
    auto proj = perspective( 45.f, float(m_rd.windowWidth/2)/float(m_rd.windowHeight), m_rd.sceneData.projNear, m_rd.sceneData.projFar );

    vec4f background = vec4f( 118.f/255.f, 185.f/255.f, 0.f/255.f, 0.f/255.f );

    // calculate some coordinate systems
    auto view           = m_control.m_viewMatrix;
    auto iview          = invert(view);
    vec3f eyePos_world  = vec3f( iview(0,3), iview(1,3), iview(2,3) );
    vec3f eyePos_view   = view * vec4f(eyePos_world, 1);
    vec3f right_view    = vec3f( 1.0f, 0.0f,  0.0f );
    vec3f up_view       = vec3f( 0.0f, 1.0f,  0.0f );
    vec3f forward_view  = vec3f( 0.0f, 0.0f, -1.0f );
    vec3f right_world   = iview * vec4f(right_view, 0.0f);
    vec3f up_world      = iview * vec4f(up_view, 0.0f);
    vec3f forward_world = iview * vec4f(forward_view, 0.0f);

    // fill sceneData struct
    m_rd.sceneData.viewMatrix     = view;
    m_rd.sceneData.projMatrix     = proj;
    m_rd.sceneData.viewProjMatrix = proj * view;
    m_rd.sceneData.lightPos_world = eyePos_world + right_world;
    m_rd.sceneData.eyepos_world   = eyePos_world;
    m_rd.sceneData.eyePos_view    = eyePos_view;
    m_rd.sceneData.color          = background;
    m_rd.sceneData.loadFactor     = m_rd.m_uiData.m_loadFactor;
    m_rd.sceneData.objectColor    = vec3f(0.75f);

    // upload scene data with gray color to both GPUs with a normal buffer upload
    // this is not really necessary, but shows gray should the multicast buffer uploads fail
    glNamedBufferSubData( m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );

    // bind scene data UBO
    glBindBufferBase(GL_UNIFORM_BUFFER, UBO_SCENE, m_rd.buf.sceneUbo);

    // prepare an FBO to render into
    glBindFramebuffer( GL_FRAMEBUFFER, m_rd.renderFBO );
    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_rd.tex.depthTex, 0 );

    float depth = 1.0f;
    glViewport(0, 0, m_rd.texWidth, m_rd.texHeight);
    glUseProgram( m_rd.pm.get( m_rd.prog.scene ) );


    // GL_NV_gpu_multicast
    if (m_rd.m_uiData.m_multicast)
    {
      // render into left texture on both GPUs, generating both images with one render call
      // then copy the left texture of GPU1 into the right texture on GPU0

      // set & upload scene data with blue object color for GPU0
      m_rd.sceneData.objectColor = vec3f(0.0f, 0.0f, 1.0f);
      glMulticastBufferSubDataNV( GPUMASK_0, m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );
      // set & upload scene data with red object color for GPU1
      m_rd.sceneData.objectColor = vec3f(1.0f, 0.0f, 0.0f);
      glMulticastBufferSubDataNV( GPUMASK_1, m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );

      // use left texture as render target
      glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_rd.tex.colorTexLeft, 0 );
      glClearBufferfv( GL_COLOR, 0, &background[0] );
      glClearBufferfv( GL_DEPTH, 0, &depth );

      // render tori into left texture
      renderTori( m_rd, m_rd.m_uiData.m_loadFactor, m_rd.windowWidth/2, m_rd.windowHeight, view );

      if (m_rd.m_uiData.m_multicast && m_rd.m_uiData.m_multicastCopy)
      {
        // this interlock makes sure colorTexRight is not still used by the previous frame's draw
        // this is unlikely if the swap chain is long enough and might be impossible depending on other 
        // syncs in the frame, so it can get skipped here.
        // glMulticastBarrierNV();
        
        // copy the left texture on GPU 1 to the right texture on GPU 0
        glMulticastCopyImageSubDataNV(1, GPUMASK_0, m_rd.tex.colorTexLeft, GL_TEXTURE_2D, 0, 0, 0, 0, m_rd.tex.colorTexRight, GL_TEXTURE_2D, 0, 0, 0, 0, m_rd.texWidth, m_rd.texHeight, 1);
        
        // Let GPU 0 wait for the copy from GPU 1: this can be done by a barrier or by an explicit sync.
        // Note that the spec ensures that the copy is synced with the source GPU automatically
        glMulticastWaitSyncNV( 1, GPUMASK_0 );
      }
    }
    else
    {
      // render into left and right texture sequentially

      // set & upload scene data with blue object color for the first image
      m_rd.sceneData.objectColor = vec3f(0.0f, 0.0f, 1.0f);
      glNamedBufferSubData( m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );

      // use left texture as render target
      glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_rd.tex.colorTexLeft, 0 );
      glClearBufferfv( GL_COLOR, 0, &background[0] );
      glClearBufferfv( GL_DEPTH, 0, &depth );

      // render tori into left texture
      renderTori( m_rd, m_rd.m_uiData.m_loadFactor, m_rd.windowWidth/2, m_rd.windowHeight, view );

      // set & upload scene data with red object color for second image
      m_rd.sceneData.objectColor = vec3f(1.0f, 0.0f, 0.0f);
      glNamedBufferSubData( m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );

      // use right texture as render target
      glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_rd.tex.colorTexRight, 0 );
      glClearBufferfv( GL_COLOR, 0, &background[0] );
      glClearBufferfv( GL_DEPTH, 0, &depth );

      // render tori into right texture
      renderTori( m_rd, m_rd.m_uiData.m_loadFactor, m_rd.windowWidth/2, m_rd.windowHeight, view );
    }

    // At this point we have a blue render in the left and a red render in the right texture
    // independent of the rendering technique unless the VR SLI copy will get implemented
    // by a framebuffer blit at the end. In that case, both GPUs still hold the rendering in 
    // the "left" texture. In this case both GPUs have to render the offscreen texture to the 
    // framebuffer to make the blit possible, otherwise this render pass can get omitted 
    // on GPU 1, which is done by setting the rendermask below:
    if(m_rd.m_uiData.m_multicast && !m_rd.m_uiData.m_multicastBlit )
    {
      // The next draw commands (combining the textures and the UI) are only needed to get performed on GPU 0:
      glRenderGpuMaskNV( GPUMASK_0 );
    }

    // render into backbuffer
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    
    // render complete viewport
    glViewport(0, 0, m_rd.windowWidth, m_rd.windowHeight);
    glUseProgram( m_rd.pm.get( m_rd.prog.compose ) );
    
    // set & upload compose data 
    m_rd.composeData.out_width  = m_rd.windowWidth;
    m_rd.composeData.out_height = m_rd.windowHeight;
    m_rd.composeData.in_width   = m_rd.texWidth;
    m_rd.composeData.in_height  = m_rd.texHeight;
    glNamedBufferSubData( m_rd.buf.composeUbo, 0, sizeof(ComposeData), &m_rd.composeData );
    glBindBufferBase( GL_UNIFORM_BUFFER, UBO_COMP, m_rd.buf.composeUbo );

    // use rendered textures as input textures
    nvgl::bindMultiTexture(GL_TEXTURE0 + 0, GL_TEXTURE_2D, m_rd.tex.colorTexLeft);
    nvgl::bindMultiTexture(GL_TEXTURE0 + 1, GL_TEXTURE_2D, m_rd.tex.colorTexRight);
    
    glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    // render one triangle covering the whole viewport
    glDrawArrays( GL_TRIANGLES, 0, 3 );

    if (m_rd.m_uiData.m_multicast && m_rd.m_uiData.m_multicastBlit)
    {
      // blit from GPU 1 to 0
      // It's an alternative to copying the textures if the final result should get merged.
      // source image is left on GPU 1 but should be right on GPU 0
      // In a blit the target GPU copies the data from the source, so the target has to wait
      // explicitly for the source to make sure everything was rendered to it:
      glMulticastWaitSyncNV( 1, GPUMASK_0 );

      glMulticastBlitFramebufferNV( 1, 0, 
        0,                  0, m_rd.windowWidth/2, m_rd.windowHeight,
        m_rd.windowWidth/2, 0, m_rd.windowWidth,   m_rd.windowHeight,
        GL_COLOR_BUFFER_BIT, GL_NEAREST );

      // Let GPU 1 wait for the blit to GPU 0: this can be done by a barrier:
      //  glMulticastBarrierNV();
      // or by an explicit sync:
      //  glMulticastWaitSyncNV( 0, GPUMASK_1 );
      // *but* as in this case the next draw commands (the UI) are only 
      // useful on GPU 0, rendering on all other GPUs gets deactivated and the
      // sync can wait until the rendering gets reactivated on those GPUs!
      glRenderGpuMaskNV( GPUMASK_0 );
    }

    if (m_rd.m_uiData.m_drawUI)
    {
      NV_PROFILE_GL_SECTION("draw ui");
      ImGui::Render();
      ImGui::RenderDrawDataGL(ImGui::GetDrawData());
    }

    ImGui::EndFrame();

    if (m_rd.m_uiData.m_multicast)
    {
      glMulticastWaitSyncNV( 0, GPUMASK_1 );
      glRenderGpuMaskNV( GPUMASK_0 | GPUMASK_1 );
    }
  }

  void Sample::resize(int width, int height)
  {
    m_windowState.m_winSize[0] = width;
    m_windowState.m_winSize[1] = height;

    m_rd.windowWidth = width;
    m_rd.windowHeight = height;
    m_rd.texWidth = width/2;
    m_rd.texHeight = height;

    initTextures( m_rd );
  }

  void Sample::end()
  {
    nvgl::deleteBuffer( m_rd.buf.vbo );
    nvgl::deleteBuffer( m_rd.buf.ibo );
    nvgl::deleteBuffer( m_rd.buf.sceneUbo );
    nvgl::deleteBuffer( m_rd.buf.objectUbo );
    nvgl::deleteBuffer( m_rd.buf.composeUbo );

    nvgl::deleteTexture( m_rd.tex.colorTexLeft );
    nvgl::deleteTexture( m_rd.tex.colorTexRight );
    nvgl::deleteTexture( m_rd.tex.depthTex );

    m_rd.pm.deletePrograms();

    nvgl::deleteFramebuffer( m_rd.renderFBO );
    nvgl::deleteFramebuffer( m_rd.tempFBO );
  }
}//namespace

int main(int argc, const char** argv)
{
  NVPSystem system(argv[0], PROJECT_NAME);

  vertexload::Sample sample;
  return sample.run(
    PROJECT_NAME,
    argc, argv,
    SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT);
}

