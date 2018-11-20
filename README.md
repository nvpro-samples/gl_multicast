# gl_multicast
In this sample the **GL_NVX_linked_gpu_multicast** extension is used to render a basic stereo scene on two GPUs in an SLI system.

> |

> |

> |

> NOTE: This sample works with drivers 361.43 (http://www.nvidia.com/download/driverResults.aspx/96888/en-us) and above.

> |

> |

> |

![sample screenshot](https://github.com/nvpro-samples/gl_multicast/blob/master/doc/sample.png)

Applications rendering stereo content, be it for display on an HMD or on another stereo display, need to generate two images per frame. These left and right eye renderings show an identical scene rendered from two different viewpoints.

With the new **GL_NVX_linked_gpu_multicast** extension it is possible to control multiple GPUs that are in an NVIDIA SLI group with a single OpenGL context to reduce overhead and improve frame rates.

While the driver is in this new multicast mode, all OpenGL operations that are not related to the extension are broadcast to all GPUs in the SLI group. Allocations are mirrored, data uploads are broadcast and rendering commands are executed on all GPUs. This means that without use of the special functionality in the new multicast extension, all GPUs are executing the same work on the same data.

For stereo rendering of a frame in VR, the GPU must render the same scene from two different eye positions. A normal application using only one GPU must render these two images sequentially, which means twice the CPU and GPU workload.

With the new OpenGL multicast extension it is possible to upload the same scene to two different GPUs and render this scene from two different viewpoints with one OpenGL rendering stream. This distributes the rendering workload across two GPUs and eliminates the CPU overhead of sending the rendering commands twice.


#### Performance
Due to the fact that for a sequential rendering of the left and the right eye almost exactly the same rendering workload is executed twice, using two GPUs with the multicast extension is beneficial in almost any cases: CPU limited renderers don't have to traverse the scene twice and generate the OpenGL stream twice. Fragment or geometry bound renderers can benefit from the additional rendering horsepower. 

The following graph shows the render times in ms and scaling for the gl_multicast sample. The sample is vertex bound, and the varying workload (on the x-axis) is realized by changing the number of objects that are rendered.

Frame times are linear with the number of objects, both for 1 and 2 GPUs, and because of the fixed overhead due to multi-GPU rendering (the difference between the frame times for 2 GPUs and perfect 2x scaling) the scaling factor asymptotically approaches the perfect value of 2.0x.

![performance graphs](https://github.com/nvpro-samples/gl_multicast/blob/master/doc/performance.png)

#### Sample Highlights
The user can change the graphics workload by either pressing the up and down arrow keys or by modifying the workload factor value.
Use of the multicast extension for rendering the image can be en- and disabled by either pressing the space bar or (un-)checking the multicast checkbox.

By searching the file main.cpp for the keyword "MULTICAST" the parts of the source code where a single-GPU application needs to be modified for multicast use can be found. Key functionality is found in the Sample::think() function, where the rendering is either done in sequence for the right and left eye or done for both eyes on two GPUs at the same time.

The following abstracted samples show the differences between a normal application doing sequential stereo rendering and an application doing parallel rendering using the new multicast extension. Since all “normal” OpenGL operations are mirrored in multicast mode, the application’s code for preparing the scene for rendering can stay unchanged, minimizing code changes when modifying an application for use of the multicast extension. 

The first sample shows how a standard application typically renders a stereo image. First, data is uploaded to the GPU containing data for the left eye (1.), and the first image is rendered (2.). Then, data for the right eye is uploaded (3.), followed by rendering the second image (4.). Calling the rendering function twice means that the OpenGL stream for rendering the image needs to be generated twice, causing twice the CPU overhead compared to the second sample.

``` cpp
// render into left and right texture sequentially

// 1. set & upload left eye for the first image
sceneData.view = /* left eye data */;
glNamedBufferSubDataEXT( ..., &sceneData );

// use left texture as render target
glFramebufferTexture2D( ..., colorTexLeft, ... );
 
// 2. render into left texture
render( ... );
 
// 3. set & upload right eye for second image
sceneData.view = /* right eye data */;
glNamedBufferSubDataEXT( ..., &sceneData );
 
// use right texture as render target
glFramebufferTexture2D( ..., colorTexRight, ... );
 
// 4. render into right texture
render( ... );
```
The second sample shows how to generate the same set of textures using two GPUs. Both data for the left and the right eye are uploaded to GPU 0 and GPU 1, respectively (1.). Since the scene data is identical on both GPUs, the single render call (2.) that gets broadcast to both GPUs generates both the left and the right eye, on GPUs 0 and 1, in their version of the left texture. To work with both textures on one GPU afterwards, the right eye image needs to be copied from GPU 1 into GPU 0 (3.).

``` cpp
// render into left texture on both GPUs
// copy left tex on GPU1 to right tex on GPU0
 
// 1. set & upload colors for both images
sceneData.view = /* left eye data */;
glLGPUNamedBufferSubDataNVX( GPUMASK_0, ..., &sceneData );
sceneData.view = /* right eye data */;
glLGPUNamedBufferSubDataNVX( GPUMASK_1, ..., &sceneData );
 
// use left texture as render target
glFramebufferTexture2D( ..., colorTexLeft, ... );
 
// 2. render into left texture
render( ... );
 
// make sure colorTexRight is safe to write
glLGPUInterlockNVX();
 
// 3. copy the left texture on GPU 1 to the right texture on GPU 0
glLGPUCopyImageSubDataNVX(
  1,         /* from GPU 1 */ 
  GPUMASK_0, /*   to GPU 0 */ 
  colorTexLeft,  GL_TEXTURE_2D, 0, 0, 0, 0,
  colorTexRight, GL_TEXTURE_2D, 0, 0, 0, 0,
  texWidth, texHeight, 1);

// make sure colorTexRight is safe to read
glLGPUInterlockNVX();
```

After both these code snippets texture 1 contains the left and texture 2 contains the right eye rendering of the scene. Since both textures reside on the GPU that the application also uses in non-multicast mode to display or pass into an HMD runtime, the code after generating the images can stay unmodified, too. As mentioned before, the simplicity of the extension requires only minimal code changes to modify an existing stereo application for multicast mode.

Note that the above code is suitable for stereo rendering, the current sample only renders the left and the right half of a image, showing the same scene in different colors. The sample is meant to showcase the multicast extension as simply as possible. Samples for actual stereo content are planned for the near future.

#### Building
Ideally clone this and other interesting [nvpro-samples](https://github.com/nvpro-samples) repositories into a common subdirectory. You will always need [shared_sources](https://github.com/nvpro-samples/shared_sources) and on Windows [shared_external](https://github.com/nvpro-samples/shared_external). The shared directories are searched either as subdirectory of the sample or one directory up.

If you are interested in multiple samples, you can use [build_all](https://github.com/nvpro-samples/build_all) CMAKE as entry point, it will also give you options to enable/disable individual samples when creating the solutions.