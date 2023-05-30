#version 430

#extension GL_ARB_shading_language_include : enable
#include "common.h"

// inputs in view space
in Interpolants{
  vec3 normal;
  vec3 eyeDir;
  vec3 lightDir;
} IN;

layout(location = 0, index = 0) out vec4 out_Color;

void main()
{
	// interpolated inputs in view space
	vec3 normal = normalize(IN.normal);
	vec3 eyeDir = normalize(IN.eyeDir);
	vec3 lightDir = normalize(IN.lightDir);

	vec3 objCol = scene.objectColor;

	// ambient term
	vec4 ambient_color = vec4(objCol * scene.color * 0.15, 1.0);

	// diffuse term
	float diffuse_intensity = max(dot(normal, lightDir), 0.0);
	vec4  diffuse_color = diffuse_intensity * vec4(objCol, 1.0);

	// specular term
	vec3  R = reflect(-lightDir, normal);
	float specular_intensity = max(dot(eyeDir, R), 0.0);
	vec4  specular_color = pow(specular_intensity, 4) * vec4(0.8, 0.8, 0.8, 1);

	out_Color = ambient_color + diffuse_color + specular_color;
}

/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
