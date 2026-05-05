use solver::{Solver, num_complex::Complex64};
use std::cell::RefCell;
use std::fs;

// 3 for plain stress
// 4 for plain strain
const NTENS: usize = 3;

thread_local! {
    static RUNNER: RefCell<Solver> = RefCell::new(Solver::new("C:\\Users\\LL_uvz\\Desktop\\project\\fft-mechanics/data/mat.png"));
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
    let strain = [
        stran[0] + dstran[0],
        stran[1] + dstran[1],
        (stran[2] + dstran[2]) * 0.5,
    ];
    RUNNER.with_borrow_mut(|runner| {
        runner.init(&strain);
        for _ in 0..1 {
            runner.step();
        }
        runner.set_ddsdde(ddsdde);
        runner.set_average_stress(stress);
        if noel == 1 && npt == 1 {
            println!("{}", runner.error());
        }
        if step == 4 && noel == 11 {
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