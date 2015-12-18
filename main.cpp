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
/* Contact iesser@nvidia.com (Ingo Esser) for feedback */

#define NVGLF_DEBUG_FILTER     1

#include <gl/glew.h>
#include <windows.h>

#include <nv_helpers/anttweakbar.hpp>
#include <nv_helpers/cameracontrol.hpp>
#include <nv_helpers/geometry.hpp>
#include <nv_helpers_gl/glresources.hpp>
#include <nv_helpers_gl/programmanager.hpp>
#include <nv_helpers_gl/WindowProfiler.hpp>
#include <nv_math/nv_math_glsltypes.h>

#include <chrono>
#include <iostream>
#include <locale>
#include <thread>

using namespace nv_math;
#include "common.h"

namespace
{
  int const SAMPLE_SIZE_WIDTH(800);
  int const SAMPLE_SIZE_HEIGHT(600);

  int const SAMPLE_MAJOR_VERSION(4);
  int const SAMPLE_MINOR_VERSION(2);


  // MULTICAST
  // typedefs for the extension functions
  typedef void (GLAPIENTRY * PFNGLLGPUNAMEDBUFFERSUBDATANVXPROC) (GLbitfield gpuMask, GLuint buffer, GLintptr offset, GLsizeiptr size, const GLvoid *data);
  typedef void (GLAPIENTRY * PFNGLLGPUCOPYIMAGESUBDATANVXPROC) (GLuint sourceGpu, GLbitfield destinationGpuMask, GLuint srcName, GLuint srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName, GLuint dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei width, GLsizei height, GLsizei depth);
  typedef void (GLAPIENTRY * PFNGLLGPUINTERLOCKNVXPROC) (void);

  // function pointers for the extension functions
  PFNGLLGPUNAMEDBUFFERSUBDATANVXPROC glLGPUNamedBufferSubDataNVX(nullptr);
  PFNGLLGPUCOPYIMAGESUBDATANVXPROC glLGPUCopyImageSubDataNVX(nullptr);
  PFNGLLGPUINTERLOCKNVXPROC glLGPUInterlockNVX(nullptr);

  // defines for the extension 
#ifndef LGPU_SEPARATE_STORAGE_BIT_NVX
#define LGPU_SEPARATE_STORAGE_BIT_NVX 0x0800
#endif
#ifndef GL_MAX_LGPU_GPUS_NVX
#define GL_MAX_LGPU_GPUS_NVX          0x92BA
#endif

  
}


namespace vertexload
{

  namespace render
  {

    struct Vertex {
      Vertex(const nv_helpers::geometry::Vertex& vertex){
        position  = vertex.position;
        normal    = vertex.normal;
        color     = nv_math::vec4(1.0f);
      }

      nv_math::vec4   position;
      nv_math::vec4   normal;
      nv_math::vec4   color;
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
      nv_helpers_gl::ProgramManager::ProgramID scene;
      nv_helpers_gl::ProgramManager::ProgramID compose;
    };

    struct Data
    {
      Data()
        : windowWidth( SAMPLE_SIZE_WIDTH )
        , windowHeight( SAMPLE_SIZE_HEIGHT )
        , texWidth( SAMPLE_SIZE_WIDTH/2 )
        , texHeight( SAMPLE_SIZE_HEIGHT )
      {
        sceneData.projNear = 0.01f;
        sceneData.projFar  = 100.0f;
      }

      Buffers   buf;
      Textures  tex;
      Programs  prog; 

      SceneData   sceneData;
      ObjectData  objectData;
      ComposeData composeData;

      GLuint renderFBO;
      GLuint tempFBO;

      nv_helpers_gl::ProgramManager pm;

      int numGPUs;
      int windowWidth;
      int windowHeight;
      int texWidth;
      int texHeight;
    };

    auto initPrograms( Data& rd ) -> bool
    {
      nv_helpers_gl::ProgramManager& pm = rd.pm;
      Programs& programs = rd.prog;

      bool validated(true);

      pm.addDirectory( std::string(PROJECT_NAME));
      pm.addDirectory( NVPWindow::sysExePath() + std::string(PROJECT_RELDIRECTORY));
      pm.addDirectory( std::string(PROJECT_ABSDIRECTORY));

      pm.registerInclude("common.h", "common.h");

      {
        programs.scene = pm.createProgram(
          nv_helpers_gl::ProgramManager::Definition(GL_VERTEX_SHADER,   "#define USE_SCENE_DATA", "scene.vert.glsl"),
          nv_helpers_gl::ProgramManager::Definition(GL_FRAGMENT_SHADER, "#define USE_SCENE_DATA", "scene.frag.glsl"));
      }

      {
        programs.compose = pm.createProgram(
          nv_helpers_gl::ProgramManager::Definition(GL_VERTEX_SHADER,   "#define USE_COMPOSE_DATA", "compose.vert.glsl"),
          nv_helpers_gl::ProgramManager::Definition(GL_FRAGMENT_SHADER, "#define USE_COMPOSE_DATA", "compose.frag.glsl"));
      }

      validated = pm.areProgramsValid();
      return validated;
    }

    auto initFBOs( Data& rd ) -> void
    {
      nv_helpers_gl::newFramebuffer( rd.renderFBO );
      nv_helpers_gl::newFramebuffer( rd.tempFBO );
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

        std::vector< nv_math::vec3 > vertices;
        std::vector< nv_math::vec3 > tangents;
        std::vector< nv_math::vec3 > binormals;
        std::vector< nv_math::vec3 > normals;
        std::vector< nv_math::vec2 > texcoords;
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

            vertices.push_back( nv_math::vec3( radius      *  cosPhi, 
              outerRadius *  sinTheta, 
              radius      * -sinPhi ) );

            tangents.push_back( nv_math::vec3( -sinPhi, 0.0f, -cosPhi ) );

            binormals.push_back( nv_math::vec3( cosPhi * -sinTheta,
              cosTheta, 
              sinPhi * sinTheta ) );

            normals.push_back( nv_math::vec3( cosPhi * cosTheta,
              sinTheta,  
              -sinPhi * cosTheta ) );

            texcoords.push_back( nv_math::vec2( (float) longitude / mf , (float) latitude / nf ) );
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

        nv_helpers_gl::newBuffer( buffers.vbo );
        glBindBuffer   ( GL_ARRAY_BUFFER, buffers.vbo );
        glBufferData   ( GL_ARRAY_BUFFER, sizePositionAttributeData + sizeNormalAttributeData, nullptr , GL_STATIC_DRAW );
        glBufferSubData( GL_ARRAY_BUFFER, 0                        , sizePositionAttributeData, &vertices[0] );
        glBufferSubData( GL_ARRAY_BUFFER, sizePositionAttributeData, sizeNormalAttributeData,   &normals[0] );
        glBindBuffer   ( GL_ARRAY_BUFFER, 0 );

        nv_helpers_gl::newBuffer( buffers.ibo );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, buffers.ibo );
        glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeIndexData, &indices[0], GL_STATIC_DRAW );
        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
      }


      // MULTICAST
      // need to mark UBOs LGPU_SEPARATE_STORAGE_BIT_NVX
      nv_helpers_gl::newBuffer( buffers.sceneUbo );
      glNamedBufferStorageEXT( buffers.sceneUbo, sizeof(SceneData), nullptr, GL_DYNAMIC_STORAGE_BIT|LGPU_SEPARATE_STORAGE_BIT_NVX );



      nv_helpers_gl::newBuffer( buffers.objectUbo );
      glBindBuffer( GL_UNIFORM_BUFFER, buffers.objectUbo );
      glBufferData( GL_UNIFORM_BUFFER, sizeof(ObjectData), nullptr, GL_DYNAMIC_DRAW );
      glBindBuffer( GL_UNIFORM_BUFFER, 0 );

      nv_helpers_gl::newBuffer( buffers.composeUbo );
      glBindBuffer( GL_UNIFORM_BUFFER, buffers.composeUbo );
      glBufferData( GL_UNIFORM_BUFFER, sizeof(ComposeData), nullptr, GL_DYNAMIC_DRAW );
      glBindBuffer( GL_UNIFORM_BUFFER, 0 );
    }

    auto initTextures( Data& rd ) -> void
    {
      auto newTex = [&]( GLuint& tex )
      {
        nv_helpers_gl::newTexture( tex );
        glBindTexture ( GL_TEXTURE_2D, tex );
        glTexStorage2D( GL_TEXTURE_2D, 1, GL_RGBA8, rd.texWidth, rd.texHeight );
        glBindTexture ( GL_TEXTURE_2D, 0);


        // MULTICAST
        // we need to clear the textures via a FBO once to get a P2P flag
        glBindFramebuffer( GL_FRAMEBUFFER, rd.tempFBO );
        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0 );
        glClearBufferfv( GL_COLOR, 0, &vec4f(0.0f)[0] );
        glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0 );
        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

      };

      newTex( rd.tex.colorTexLeft );
      newTex( rd.tex.colorTexRight );

      nv_helpers_gl::newTexture( rd.tex.depthTex );
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
          rd.objectData.model = nv_math::scale_mat4( nv_math::vec3(scale) ) * nv_math::translation_mat4( nv_math::vec3( x, y, 0.0f ) ) * nv_math::rotation_mat4_x( (j%2?-1.0f:1.0f) * 45.0f * nv_pi/180.0f );

          rd.objectData.modelView = view * rd.objectData.model;
          rd.objectData.modelViewIT = nv_math::transpose(nv_math::invert(rd.objectData.modelView));
          rd.objectData.modelViewProj = rd.sceneData.viewProjMatrix * rd.objectData.model;

          rd.objectData.color = nv_math::vec3f( (torusIndex+1)&1, ((torusIndex+1)&2)/2, ((torusIndex+1)&4)/4 );

          // set model UBO
          glNamedBufferSubDataEXT( rd.buf.objectUbo, 0, sizeof(ObjectData), &rd.objectData );
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

  class Sample : public nv_helpers_gl::WindowProfiler {
  public:
    Sample();

    bool begin();
    void think(double time);
    void resize(int width, int height);
    void end();

    bool mouse_pos    (int x, int y) {
      return !!TwEventMousePosGLFW(x,y); 
    }
    bool mouse_button (int button, int action) {
      return !!TwEventMouseButtonGLFW(button, action);
    }
    bool mouse_wheel  (int wheel) {
      return !!TwEventMouseWheelGLFW(wheel); 
    }
    bool key_button   (int button, int action, int mods) {
      return nv_helpers::handleTwKeyPressed(button,action,mods);
    }

  protected:
    struct Tweak {
      Tweak() 
#if BENCHMARK_MODE
        : m_loadFactor( SAMPLE_LOAD*20 )
#else 
        : m_loadFactor( 6 )
#endif
        , m_multicast( 1 )
      {}

      int  m_loadFactor;
      char m_multicast;
    };

    Tweak                     m_tweak;
    nv_helpers::CameraControl m_control;
    size_t                    m_frameCount;
    render::Data              m_rd;
  };

  Sample::Sample()
    : nv_helpers_gl::WindowProfiler( /*singleThreaded=*/true, /*doSwap=*/true ) 
    , m_frameCount( 0 )
  {


    // MULTICAST
    // export the environment variable BEFORE the first OpenGL context is initialized
    putenv("GL_NVX_LINKED_GPU_MULTICAST=1");


  }

  bool Sample::begin()
  {
    std::cout << "Setting locale to " << std::locale("").name().c_str() << "\n";
    std::locale::global(std::locale(""));
    std::cout.imbue(std::locale());

    TwInit(TW_OPENGL_CORE,NULL);
    TwWindowSize(m_window.m_viewsize[0],m_window.m_viewsize[1]);

    vsync( false );

#if BENCHMARK_MODE
    m_profilerPrint = false;
#endif

    bool validated( true );


    // MULTICAST
    // look for the extension in the extensions list
    GLint numExtensions;
    glGetIntegerv( GL_NUM_EXTENSIONS, &numExtensions );

    bool found = false;
    for( GLint i=0; i < numExtensions && !found; ++i )
    {
      std::string name( (const char*)glGetStringi( GL_EXTENSIONS, i ) );
      if( name == "GL_NVX_linked_gpu_multicast" )
      {
        std::cout << "Extension " << name << " found!\n";
        found = true;
      }
    }

    if( !found )
    {
      minimize();
      std::cerr << "Multicast extension not found, aborting!\n\n";
      std::this_thread::sleep_for( std::chrono::seconds(5) );
      return false;
    }

    // get pointers to the extension functions
    glLGPUNamedBufferSubDataNVX = (PFNGLLGPUNAMEDBUFFERSUBDATANVXPROC)wglGetProcAddress("glLGPUNamedBufferSubDataNVX");
    glLGPUCopyImageSubDataNVX = (PFNGLLGPUCOPYIMAGESUBDATANVXPROC)wglGetProcAddress("glLGPUCopyImageSubDataNVX");
    glLGPUInterlockNVX = (PFNGLLGPUINTERLOCKNVXPROC)wglGetProcAddress("glLGPUInterlockNVX");

    if(  glLGPUNamedBufferSubDataNVX == nullptr 
      || glLGPUCopyImageSubDataNVX == nullptr 
      || glLGPUInterlockNVX == nullptr )
    {
      minimize();
      std::cerr << "\n\nGL_NVX_linked_gpu_multicast not supported, aborting!\n\n";
      std::this_thread::sleep_for( std::chrono::seconds(5) );
      return false;
    }

    // query the number of GPUs available in this system
    glGetIntegerv(GL_MAX_LGPU_GPUS_NVX, &m_rd.numGPUs);
    std::cout << "GPUs found: " << m_rd.numGPUs << "\n";


    // control setup
    m_control.m_sceneOrbit = nv_math::vec3(0.0f);
    m_control.m_sceneDimension = 1.0f;
    m_control.m_viewMatrix = nv_math::look_at(m_control.m_sceneOrbit - vec3( 0, 0 ,-m_control.m_sceneDimension ), m_control.m_sceneOrbit, vec3(0,1,0));

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

    TwBar *bar = TwNewBar("mainbar");
    TwDefine(" GLOBAL contained=true help='OpenGL samples.\nCopyright NVIDIA Corporation 2014' ");
    TwDefine(" mainbar position='0 0' size='350 200' color='0 0 0' alpha=128 valueswidth=170 ");
    TwDefine((std::string(" mainbar label='") + PROJECT_NAME + "'").c_str());

    TwAddVarRW(bar, "loadFactor",  TW_TYPE_UINT16,  &m_tweak.m_loadFactor  , nullptr);
    TwAddVarRW(bar, "multicast",   TW_TYPE_BOOL8,  &m_tweak.m_multicast,    nullptr);

    return validated;
  }

  void Sample::think(double time)
  {
    // handle mouse input
    m_control.processActions(m_window.m_viewsize,
      nv_math::vec2f(m_window.m_mouseCurrent[0],m_window.m_mouseCurrent[1]),
      m_window.m_mouseButtonFlags, m_window.m_wheel);

    // handle keyboard inputs, change number of objects
    if( m_window.onPress(KEY_UP) )
    {
      m_tweak.m_loadFactor += SAMPLE_LOAD;
      std::cout << "loadFactor: " << m_tweak.m_loadFactor << std::endl;
    }
    if( m_window.onPress(KEY_DOWN) )
    {
      if( m_tweak.m_loadFactor > SAMPLE_LOAD )
      {
        m_tweak.m_loadFactor -= SAMPLE_LOAD;
      }
      std::cout << "loadFactor: " << m_tweak.m_loadFactor << std::endl;
    }
    if( m_window.onPress(KEY_SPACE) )
    {
      m_tweak.m_multicast = m_tweak.m_multicast?0:1;
      std::cout << "multicast " << (m_tweak.m_multicast?"en":"dis") << "abled\n";
    }


#if BENCHMARK_MODE
    static double timeBegin = sysGetTime();
    static double frames = 0;

    ++frames;
    double timeCurrent = sysGetTime();
    double timeDelta = timeCurrent - timeBegin;
    if( timeDelta > 5.0 )
    {
      std::cout << timeDelta*1000.0/frames << "\n";
      if( m_tweak.m_loadFactor < SAMPLE_LOAD*20 )
      {
        m_tweak.m_loadFactor += SAMPLE_LOAD;
      }
      else
      {
        std::cout << "---------\nloadfactors from " << SAMPLE_LOAD << " .. " << SAMPLE_LOAD*20 << " :\n";
        m_tweak.m_loadFactor = SAMPLE_LOAD;
      }

      frames = 0;
      timeBegin = timeCurrent;
    }
#endif

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
    m_rd.sceneData.loadFactor     = m_tweak.m_loadFactor;
    m_rd.sceneData.objectColor    = vec3f(0.75f);

    // upload scene data with gray color to both GPUs with a normal buffer upload
    // this is not really necessary, but shows gray when the multicast buffer uploads fail
    glNamedBufferSubDataEXT( m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );

    // bind scene data UBO
    glBindBufferBase(GL_UNIFORM_BUFFER, UBO_SCENE, m_rd.buf.sceneUbo);

    // prepare an FBO to render into
    glBindFramebuffer( GL_FRAMEBUFFER, m_rd.renderFBO );
    glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_rd.tex.depthTex, 0 );

    float depth = 1.0f;
    glViewport(0, 0, m_rd.texWidth, m_rd.texHeight);
    glUseProgram( m_rd.pm.get( m_rd.prog.scene ) );

    // MULTICAST
    if( m_tweak.m_multicast )
    {
      // render into left texture on both GPUs, generating both images with one render call
      // then copy the left texture of GPU1 into the right texture on GPU0

      // set & upload scene data with blue object color for GPU0
      m_rd.sceneData.objectColor = vec3f(0.0f, 0.0f, 1.0f);
      glLGPUNamedBufferSubDataNVX( GPUMASK_0, m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );
      // set & upload scene data with red object color for GPU1
      m_rd.sceneData.objectColor = vec3f(1.0f, 0.0f, 0.0f);
      glLGPUNamedBufferSubDataNVX( GPUMASK_1, m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );

      // use left texture as render target
      glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_rd.tex.colorTexLeft, 0 );
      glClearBufferfv( GL_COLOR, 0, &background[0] );
      glClearBufferfv( GL_DEPTH, 0, &depth );

      // render tori into left texture
      renderTori( m_rd, m_tweak.m_loadFactor, m_rd.windowWidth/2, m_rd.windowHeight, view );

      // this interlock makes sure colorTexRight is not still used by the previous frame's draw
      glLGPUInterlockNVX();

      // copy the left texture on GPU 1 to the right texture on GPU 0
      glLGPUCopyImageSubDataNVX(1, GPUMASK_0, m_rd.tex.colorTexLeft, GL_TEXTURE_2D, 0, 0, 0, 0, m_rd.tex.colorTexRight, GL_TEXTURE_2D, 0, 0, 0, 0, m_rd.texWidth, m_rd.texHeight, 1);

      // this interlock makes sure colorTexRight is complete and safe to use in the subsequent composition draw
      glLGPUInterlockNVX();
    }
    else
    {
      // render into left and right texture sequentially

      // set & upload scene data with blue object color for the first image
      m_rd.sceneData.objectColor = vec3f(0.0f, 0.0f, 1.0f);
      glNamedBufferSubDataEXT( m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );

      // use left texture as render target
      glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_rd.tex.colorTexLeft, 0 );
      glClearBufferfv( GL_COLOR, 0, &background[0] );
      glClearBufferfv( GL_DEPTH, 0, &depth );

      // render tori into left texture
      renderTori( m_rd, m_tweak.m_loadFactor, m_rd.windowWidth/2, m_rd.windowHeight, view );

      // set & upload scene data with red object color for second image
      m_rd.sceneData.objectColor = vec3f(1.0f, 0.0f, 0.0f);
      glNamedBufferSubDataEXT( m_rd.buf.sceneUbo, 0, sizeof(SceneData), &m_rd.sceneData );

      // use right texture as render target
      glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_rd.tex.colorTexRight, 0 );
      glClearBufferfv( GL_COLOR, 0, &background[0] );
      glClearBufferfv( GL_DEPTH, 0, &depth );

      // render tori into right texture
      renderTori( m_rd, m_tweak.m_loadFactor, m_rd.windowWidth/2, m_rd.windowHeight, view );
      }



    // at this point we have a blue render in the left and a red render in the right texture
    // independent of the rendering technique


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
    glNamedBufferSubDataEXT( m_rd.buf.composeUbo, 0, sizeof(ComposeData), &m_rd.composeData );
    glBindBufferBase( GL_UNIFORM_BUFFER, UBO_COMP, m_rd.buf.composeUbo );

    // use rendered textures as input textures
    glBindMultiTextureEXT( GL_TEXTURE0 + 0, GL_TEXTURE_2D, m_rd.tex.colorTexLeft );
    glBindMultiTextureEXT( GL_TEXTURE0 + 1, GL_TEXTURE_2D, m_rd.tex.colorTexRight );

    glClearColor( 0.0f, 0.0f, 0.0f, 1.0f );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    // render one triangle covering the whole viewport
    glDrawArrays( GL_TRIANGLES, 0, 3 );

    TwDraw();
  }

  void Sample::resize(int width, int height)
  {
    m_window.m_viewsize[0] = width;
    m_window.m_viewsize[1] = height;

    m_rd.windowWidth = width;
    m_rd.windowHeight = height;
    m_rd.texWidth = width/2;
    m_rd.texHeight = height;

    initTextures( m_rd );

    TwWindowSize(width, height);
  }

  void Sample::end()
  {
    nv_helpers_gl::deleteBuffer( m_rd.buf.vbo );
    nv_helpers_gl::deleteBuffer( m_rd.buf.ibo );
    nv_helpers_gl::deleteBuffer( m_rd.buf.sceneUbo );
    nv_helpers_gl::deleteBuffer( m_rd.buf.objectUbo );
    nv_helpers_gl::deleteBuffer( m_rd.buf.composeUbo );

    nv_helpers_gl::deleteTexture( m_rd.tex.colorTexLeft );
    nv_helpers_gl::deleteTexture( m_rd.tex.colorTexRight );
    nv_helpers_gl::deleteTexture( m_rd.tex.depthTex );

    m_rd.pm.deletePrograms();

    nv_helpers_gl::deleteFramebuffer( m_rd.renderFBO );
    nv_helpers_gl::deleteFramebuffer( m_rd.tempFBO );
  }

}//namespace

int sample_main(int argc, const char** argv)
{
  vertexload::Sample sample;
  return sample.run(
    PROJECT_NAME,
    argc, argv,
    SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT, 
    SAMPLE_MAJOR_VERSION, SAMPLE_MINOR_VERSION);
}

void sample_print(int level, const char * fmt)
{}

