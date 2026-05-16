use num_complex::Complex64;
use solver::new::{Field, Solver, field_to_real};

const CHECK_X: usize = 0;
const CHECK_Y: usize = 0;

fn main() {
    // solver::new::test();

    let mut solver = Solver::new("data/mat.png");
    solver.init(&[0.2, 0.0, 0.0]);
    for _ in 0..10 {
        solver.step();
        // let error = solver.error();
        // println!("Step: {}, Error: {}", i, error);
    }
    save_complex_img(&solver.stress()[0][0], "data/s11.png");
    println!("{}", solver.stress()[0][0].get([CHECK_X, CHECK_Y]));
    save_complex_img(&solver.stress()[1][1], "data/s22.png");
    println!("{}", solver.stress()[1][1].get([CHECK_X, CHECK_Y]));
    save_complex_img(&solver.stress()[0][1], "data/s12.png");
    println!("{}", solver.stress()[0][1].get([CHECK_X, CHECK_Y]));

    // for i in 0..2 {
    //     for j in 0..2 {
    //         print!("{}, ", solver.stress()[i][j].get([0, 0]));
    //     }
    //     println!();
    // }

    test();
}

fn save_complex_img(f: &Field<Complex64>, path: &str) {
    let f = field_to_real(f);
    f.save_img(path);
}

fn test() {
    let mut solver = solver::Solver::new("data/mat.png");
    solver.init_with_v3(&[0.2, 0.0, 0.0]);
    for _ in 0..10 {
        solver.step();
        // let error = solver.error();
        // println!("Step: {}, Error: {}", i, error);
    }
    fn save_complex_img(f: &solver::Field2d<Complex64>, path: &str) {
        let f = solver::field_to_real(f);
        f.save_img(path);
    }
    // save_complex_img(&solver.stress()[0][0], "data/s11.png");
    // println!("{}", solver.stress()[0][0].data[CHECK_X][CHECK_Y]);
    // save_complex_img(&solver.stress()[1][1], "data/s22.png");
    // println!("{}", solver.stress()[1][1].data[CHECK_X][CHECK_Y]);
    // save_complex_img(&solver.stress()[0][1], "data/s12.png");
    // println!("{}", solver.stress()[0][1].data[CHECK_X][CHECK_Y]);
}
