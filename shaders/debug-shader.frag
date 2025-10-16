#version 450

// Texture coordinates from the vertex shader
layout (location = 0) in vec2 tex_coord;

// Color from our vertex shader
layout (location = 1) in vec3 frag_color;

// Final color of the pixel
layout (location = 0) out vec4 final_color;

void main() {
	final_color = vec4(frag_color, 1.0);
	final_color.r = tex_coord.x;
	final_color.g = tex_coord.y;
}