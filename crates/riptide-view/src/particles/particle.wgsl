struct View {
    view_proj: mat4x4<f32>,
    inverse_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inverse_view: mat4x4<f32>,
    projection: mat4x4<f32>,
    inverse_projection: mat4x4<f32>,
    world_position: vec3<f32>,
    viewport: vec4<f32>,
};

struct Particle {
    model: mat4x4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> view: View;

@group(1) @binding(0)
var<uniform> particle: Particle;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
#ifdef VERTEX_COLOR
    @location(2) color: vec4<f32>,
#endif
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
#ifdef VERTEX_COLOR
    @location(1) color: vec4<f32>,
#endif
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
#ifdef LOCK_ROTATION
    let vertex_position = vec4<f32>(-vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
    let position = view.view_proj * particle.model * vertex_position;
#else
    let camera_right = normalize(vec3<f32>(view.view_proj.x.x, view.view_proj.y.x, view.view_proj.z.x));
#ifdef LOCK_Y
    let camera_up = vec3<f32>(0.0, 1.0, 0.0);
#else
    let camera_up = normalize(vec3<f32>(view.view_proj.x.y, view.view_proj.y.y, view.view_proj.z.y));
#endif // LOCK_Y

    let world_space = camera_right * vertex.position.x + camera_up * vertex.position.y;
    let position = view.view_proj * particle.model * vec4<f32>(world_space, 1.0);
#endif // LOCK_ROTATION

    var out: VertexOutput;
    out.position = position;
    out.uv = vertex.uv;

#ifdef VERTEX_COLOR
    out.color = vertex.color;
#endif

    return out;
}

struct Fragment {
    @location(0) uv: vec2<f32>,
#ifdef VERTEX_COLOR
    @location(1) color: vec4<f32>,
#endif
};

@fragment
fn fragment(fragment: Fragment) -> @location(0) vec4<f32> {
    let uv_norm = fragment.uv * 2.0 - 1.0;
    let d = dot(uv_norm, uv_norm);

    if (d > 1.0) {
        discard;
    }

    let color = particle.color;

#ifdef VERTEX_COLOR
    return color * fragment.color;
#else
    return color;
#endif
}
