use solver::Solver;
use std::cell::RefCell;

// 3 for plain stress
// 4 for plain strain
const NTENS: usize = 3;

thread_local! {
    static RUNNER: RefCell<Solver> = RefCell::new(Solver::new());
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
    STEPS.with_borrow_mut(|step| {
        if noel == 1 && npt == 1 {
            *step += 1;
            println!("step: {}", step);
        }
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
    });
}