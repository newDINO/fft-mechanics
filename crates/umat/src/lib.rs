
// 3 for plain stress
// 4 for plain strain
const NTENS: usize = 3;

#[unsafe(no_mangle)]
pub extern "C" fn run(
    ddsdde: &mut [[f64; NTENS]; NTENS],
    stress: &mut [f64; NTENS],
    stran: &[f64; NTENS],
    dstran: &[f64; NTENS],
    noel: i32,
    npt: i32,
) {
    let _ = (stran, npt, noel);

    let e = 2.0;
    let nu = 0.3;

    let a = e / (1.0 - nu * nu);
    let b = nu * a;
    let c = 0.5 * (1.0 - nu) * a;

    for i in 0..NTENS {
        for j in 0..NTENS {
            ddsdde[i][j] = 0.0;
        }
    }

    ddsdde[0][0] = a;
    ddsdde[1][1] = a;
    ddsdde[2][2] = c;
    ddsdde[0][1] = b;
    ddsdde[1][0] = b;

    for i in 0..NTENS {
        for j in 0..NTENS {
            stress[i] += ddsdde[i][j] * dstran[j];
        }
    }
}