use num_complex::{Complex, Complex64};
use solver::{Field2d, H, Solver, W};

fn main() {
    let mut solver = Solver::new("data/mat.png");
    solver.init(&[0.2, 0.0, 0.05]);
    for i in 0..10 {
        solver.step();
        let error = solver.error();
        println!("Step: {}, Error: {}", i, error);
    }
    save_complex_img(&solver.stress()[0][0], "data/s11.png");
    save_complex_img(&solver.stress()[1][1], "data/s22.png");
    save_complex_img(&solver.stress()[0][1], "data/s12.png");
}

fn save_complex_img(f: &Field2d<Complex64>, path: &str) {
    let f = field2d_to_real(f);
    f.save_img(path);
}

fn field2d_to_real<T: Copy + Default>(f: &Field2d<Complex<T>>) -> Field2d<T> {
    let mut result: Field2d<T> = Default::default();
    for i in 0..W {
        for j in 0..H {
            result.data[i][j] = f.data[i][j].re;
        }
    }
    result
}
