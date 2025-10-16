#version 450

// Get the vertex position from the vertex buffer
layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 tex_coord;

// Output texture coordinates to the fragment shader
layout (location = 0) out vec2 out_tex_coord;

// Output a color to the fragment shader
layout (location = 1) out vec3 frag_color;

// Generates an orthographic projection matrix
mat4 ortho(float left, float right, float bottom, float top, float near, float far) {
    return mat4(
        2.0 / (right - left), 0, 0, 0,
        0, 2.0 / (top - bottom), 0, 0,
        // Note: this is assuming a clip space of [0, 1] on the Z axis, which is what Vulkan uses.
        // In OpenGL, the clip space is [-1, 1] and this would need to be adjusted.
        0, 0, -1.0 / (far - near), 0,
        -(right + left) / (right - left), -(top + bottom) / (top - bottom), -near / (far - near), 1
    );
}

// Generates a simple isometric view matrix since the program isn't
// passing in a uniform view matrix. Without this, we'd just see the
// front side of the cube and nothing else.
mat4 isometric_view_matrix() {
    return mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    );
}

void main(void) {
    // Calculate the final vertex position by multiplying in the projection and view matrices.
    // Ordinarily, these matrices would be passed in as uniforms, but here they're
    // being calculated in-shader to avoid pulling in a matrix multiplication library.
    mat4 proj_matrix = ortho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
    mat4 view_matrix = isometric_view_matrix();
	gl_Position = proj_matrix * view_matrix * vec4(pos, 1.0);
    out_tex_coord = tex_coord;

    // Create a frag color based on the vertex position
    frag_color = normalize(pos) * 0.5 + 0.5;
}