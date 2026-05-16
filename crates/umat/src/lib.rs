use solver::{Solver, num_complex::Complex64};
use std::cell::RefCell;
use std::fs;

// 3 for plain stress
// 4 for plain strain
const NTENS: usize = 3;

const MAT_IMAGE_PATH: &str = "C:\\Users\\LL_uvz\\Desktop\\project\\fft-mechanics/data/mat.png";

thread_local! {
    static RUNNER: RefCell<Solver> = RefCell::new(Solver::new(MAT_IMAGE_PATH));
    static STEPS: RefCell<u32> = Default::default();
}

#[unsafe(no_mangle)]
pub extern "C" fn run(
    ddsdde: &mut [[f64; NTENS]; NTENS],
    stress: &mut [f64; NTENS],
    stran: &[f64; NTENS],
    dstran: &[f64; NTENS],
    noel: i32,
    npt: i32,
) {
    let step = STEPS.with_borrow_mut(|step| {
        if noel == 1 && npt == 1 {
            *step += 1;
            println!("step: {}", step);
        }
        *step
    });
    let strain = abq_strain_to_t22(stran, dstran);
    RUNNER.with_borrow_mut(|runner| {
        runner.init(strain);
        for _ in 0..1 {
            runner.step();
        }
        runner.set_ddsdde(ddsdde);

        let new_stress = runner.get_average_stress();
        *stress = t22_to_v3(new_stress);

        if noel == 1 && npt == 1 {
            println!("{}", runner.error());
        }
        if step == 3 && noel == 1 {
            let stress = runner.stress();
            const PATH: &str = "C:\\Users\\LL_uvz\\Desktop\\project\\fft-mechanics";
            save_field_as_real(&format!("{}/data/output/pt{}s11", PATH, npt), &stress[0][0]);
            save_field_as_real(&format!("{}/data/output/pt{}s22", PATH, npt), &stress[1][1]);
            save_field_as_real(&format!("{}/data/output/pt{}s12", PATH, npt), &stress[0][1]);
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn run_with_rotation(
    ddsdde: &mut [[f64; NTENS]; NTENS],
    stress: &mut [f64; NTENS],
    stran: &[f64; NTENS],
    dstran: &[f64; NTENS],
    coords: &[f64; 3],
    noel: i32,
    npt: i32,
) {
    let coords_len = (coords[0] * coords[0] + coords[1] * coords[1]).sqrt();
    let coords = [coords[0] / coords_len, coords[1] / coords_len];
    let rotation = [
        [coords[1], -coords[0]],
        [coords[0], coords[1]],
    ];
    let rotation_t = [
        [coords[1], coords[0]],
        [-coords[0], coords[1]],
    ];

    let step = STEPS.with_borrow_mut(|step| {
        if noel == 1 && npt == 1 {
            *step += 1;
            println!("SIMD step: {}", step);
        }
        *step
    });
    let strain0 = abq_strain_to_t22(stran, dstran);
    let strain = mat_mul(mat_mul(rotation_t, strain0), rotation);

    RUNNER.with_borrow_mut(|runner| {
        runner.init(strain);
        for _ in 0..1 {
            runner.step();
        }
        runner.set_ddsdde(ddsdde);

        let new_stress = runner.get_average_stress();
        let new_stress0 = mat_mul(mat_mul(rotation, new_stress), rotation_t);
        *stress = t22_to_v3(new_stress0);

        if noel == 1 && npt == 1 {
            println!("{}", runner.error());
        }
        if step == 5 && noel == 5 {
            let stress = runner.stress();
            const PATH: &str = "C:\\Users\\LL_uvz\\Desktop\\project\\fft-mechanics";
            save_field_as_real(&format!("{}/data/output/pt{}s11", PATH, npt), &stress[0][0]);
            save_field_as_real(&format!("{}/data/output/pt{}s22", PATH, npt), &stress[1][1]);
            save_field_as_real(&format!("{}/data/output/pt{}s12", PATH, npt), &stress[0][1]);
        }
    });
}

fn save_field_as_real(path: &str, field: &solver::Field2d<Complex64>) {
    let real = solver::field_to_real(field);
    if let Err(e) = fs::write(path, bytemuck::cast_slice(real.data.as_slice())) {
        println!("Error: {:?}\nPath: {}", e, path);
    };
}

type T22 = [[f64; 2]; 2];

fn abq_strain_to_t22(stran: &[f64; 3], dstran: &[f64; 3]) -> T22 {
    let strain = [
        stran[0] + dstran[0],
        stran[1] + dstran[1],
        (stran[2] + dstran[2]) * 0.5,
    ];
    v3_to_t22(strain)
}

fn v3_to_t22(v: [f64; 3]) -> T22 {
    [
        [v[0], v[2]],
        [v[2], v[1]],
    ]
}
fn t22_to_v3(t: T22) -> [f64; 3] {
    [t[0][0], t[1][1], t[0][1]]
}

fn mat_mul(m1: T22, m2: T22) -> T22 {
    [
        [m1[0][0] * m2[0][0] + m1[0][1] * m2[1][0], m1[0][0] * m2[0][1] + m1[0][1] * m2[1][1]],
        [m1[1][0] * m2[0][0] + m1[1][1] * m2[1][0], m1[1][0] * m2[0][1] + m1[1][1] * m2[1][1]],
    ]
}