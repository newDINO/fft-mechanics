use glam::{Vec3, vec3};
use image::GrayImage;

fn main() {
    let (mut models, _) = tobj::load_obj("data/test.obj", &tobj::GPU_LOAD_OPTIONS).unwrap();
    let model = models.pop().unwrap();
    let mesh = model.mesh;

    let min_corners = mesh
        .positions
        .chunks(3)
        .fold(Vec3::splat(f32::INFINITY), |min, pos| {
            min.min(Vec3::from_slice(pos))
        });
    let min_corners = min_corners - Vec3::splat(0.01);
    let voxel_size = 2.0 / 16.0;

    let size = glam::usizevec3(16, 16, 16);
    let area = size.x * size.y;
    let mut voxels: Vec<u8> = vec![0; size.x * size.y * size.z];

    for z in 0..size.z {
        let z_base = z * area;
        for y in 0..size.y {
            let yz_base = z_base + y * size.x;

            for x in 0..size.x {
                let index = yz_base + x;

                let voxel_pos = min_corners + Vec3::new(x as f32, y as f32, z as f32) * voxel_size;

                let ray_dir = vec3(1.0, 0.0, 0.0);
                let mut hit_count = 0;

                for triangle in mesh.indices.chunks(3) {
                    let a = Vec3::from_slice(&mesh.positions[triangle[0] as usize * 3..]);
                    let b = Vec3::from_slice(&mesh.positions[triangle[1] as usize * 3..]);
                    let c = Vec3::from_slice(&mesh.positions[triangle[2] as usize * 3..]);

                    if let Some(_) = ray_intersect_triangle(voxel_pos, ray_dir, a, b, c) {
                        hit_count += 1;
                    }
                }

                voxels[index as usize] = if hit_count % 2 == 1 { 255 } else { 0 };
            }
        }
    }

    for z in 0..size.z {
        let image = GrayImage::from_raw(
            size.x as u32,
            size.y as u32,
            voxels[(z * area) as usize..((z + 1) * area) as usize].to_vec(),
        )
        .unwrap();
        image
            .save(format!("data/images/voxels/slice_{}.png", z))
            .unwrap();
    }
}

// fn point_in_triangle(point: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
//     let v0 = c - a;
//     let v1 = b - a;
//     let v2 = point - a;

//     let dot00 = v0.dot(v0);
//     let dot01 = v0.dot(v1);
//     let dot02 = v0.dot(v2);
//     let dot11 = v1.dot(v1);
//     let dot12 = v1.dot(v2);

//     let denom = dot00 * dot11 - dot01 * dot01;
//     if denom.abs() < f32::EPSILON {
//         return false;
//     }

//     let inv_denom = 1.0 / denom;
//     let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
//     let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

//     u >= 0.0 && v >= 0.0 && u + v <= 1.0
// }

fn ray_intersect_triangle(
    ray_origin: Vec3,
    ray_dir: Vec3,
    a: Vec3,
    b: Vec3,
    c: Vec3,
) -> Option<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let h = ray_dir.cross(edge2);
    let det = edge1.dot(h);

    if det.abs() < f32::EPSILON {
        return None; // Ray is parallel to triangle
    }

    let inv_det = 1.0 / det;
    let s = ray_origin - a;
    let u = s.dot(h) * inv_det;

    if u < 0.0 || u > 1.0 {
        return None; // Intersection is outside the triangle
    }

    let q = s.cross(edge1);
    let v = ray_dir.dot(q) * inv_det;

    if v < 0.0 || u + v > 1.0 {
        return None; // Intersection is outside the triangle
    }

    let t = edge2.dot(q) * inv_det;
    if t > f32::EPSILON {
        Some(t) // Intersection at distance t along the ray
    } else {
        None // Line intersection but not a ray intersection
    }
}
