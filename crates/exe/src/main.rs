use std::fs;

use solver::new::{Solver, field_to_real, for_in_t2};

fn main() {
    let mut solver = Solver::new("data/mat.png");
    // solver.init([[0.2, 0.0], [0.0, 0.0]]);
    solver.init(&[0.2, 0.0, 0.0, 0.0, 0.0, 0.5]);
    for _ in 0..1 {
        solver.step();
    }

    // for_in_t2(|a, b| {
    //     // fs::write(
    //     //     format!("data/output/{}{}", a, b),
    //     //     bytemuck::cast_slice(field_to_real(&solver.stress()[a][b]).data.as_slice()),
    //     // )
    //     // .unwrap()
    //     println!("{a}{b}: {}", solver.stress()[a][b].get([0, 0, 0]).re);
    // });
}

// fn main() {
//     let mut solver = Solver::new("data/mat.png");
//     test();

//     // println!("{}", solver::new::TEST_FIELD == solver::TEST_FIELD);
// }

// fn test() {
//     let mut solver = solver::Solver::new("data/mat.png");
// }
