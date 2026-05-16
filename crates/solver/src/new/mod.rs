use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex64};

mod field;
use field::*;
pub use field::{Field, H, W, field_to_real, for_in_t2};

mod mechanics;
use mechanics::*;

#[cfg(feature = "d2")]
const NTENS: usize = 3;
#[cfg(feature = "d3")]
const NTENS: usize = 6;

#[cfg(feature = "d2")]
const INV_N: f64 = 1.0 / (W * H) as f64;
#[cfg(feature = "d3")]
const INV_N: f64 = 1.0 / (W * H * D) as f64;

pub struct Solver {
    ffts: [Arc<dyn Fft<f64>>; DIM],
    iffts: [Arc<dyn Fft<f64>>; DIM],
    fft_scratch: [Complex64; W],

    freqs: [Field<f64>; DIM],

    greens: Tensor4<f64>,
    c: Tensor4<f64>,
    ddsdde: [[f64; NTENS]; NTENS],

    strain0: [[f64; DIM]; DIM],

    stress: Tensor2<Complex64>,
    strain: Tensor2<Complex64>,
    stress_fft: Tensor2<Complex64>,
    strain_fft: Tensor2<Complex64>,
}

impl Solver {
    pub fn new(mat_img_path: &str) -> Self {
        let mut planner = FftPlanner::<f64>::new();

        let shape = [
            W,
            H,
            #[cfg(feature = "d3")]
            D,
        ];
        let ffts = shape.map(|l| planner.plan_fft_forward(l));
        let iffts = shape.map(|l| planner.plan_fft_inverse(l));
        let fft_scratch = [Complex64::default(); W];

        let (e1, e2) = (1.0, 2.0);
        let nu = 0.3;
        #[cfg(feature = "d2")]
        let is_plain_stress = false;
        #[cfg(feature = "d3")]
        let is_plain_stress = false;

        let lame1 = get_lame(e1, nu, is_plain_stress);
        let lame2 = get_lame(e2, nu, is_plain_stress);
        let mu1 = get_shear_modulus(e1, nu);
        let mu2 = get_shear_modulus(e2, nu);

        let freqs = get_freqs();

        let greens = init_greens(lame1, mu1, &freqs);

        let (c, ddsdde) = {
            use image::ImageReader;
            let img = ImageReader::open(mat_img_path)
                .unwrap()
                .decode()
                .unwrap()
                .into_luma8();
            let c = init_c_from_image(lame1, lame2, mu1, mu2, &img);
            // let c = init_ordinary_c(lame1, mu1);
            let ddsdde = init_ddsdde_from_image(lame1, lame2, mu1, mu2, &img);
            (c, ddsdde)
        };

        Self {
            ffts,
            iffts,
            fft_scratch,

            freqs,

            greens,
            c,
            ddsdde,

            strain0: Default::default(),

            stress: Default::default(),
            strain: Default::default(),
            stress_fft: Default::default(),
            strain_fft: Default::default(),
        }
    }
    pub fn init(&mut self, strain: &[f64; NTENS]) {
        #[cfg(feature = "d2")]
        {
            self.strain0[0][0] = strain[0];
            self.strain0[1][1] = strain[1];
            self.strain0[0][1] = strain[2];
            self.strain0[1][0] = strain[2];
        }
        #[cfg(feature = "d3")]
        {
            self.strain0[0][0] = strain[0];
            self.strain0[1][1] = strain[1];
            self.strain0[2][2] = strain[2];
            self.strain0[1][0] = strain[3];
            self.strain0[0][1] = strain[3];
            self.strain0[2][0] = strain[4];
            self.strain0[0][2] = strain[4];
            self.strain0[2][1] = strain[5];
            self.strain0[1][2] = strain[5];
        }

        for_in_t2(|i, j| {
            for_in_field(|index| self.strain[i][j].set(index, self.strain0[i][j].into()))
        });

        tensor_mul_assign(&mut self.stress, &self.strain, &self.c);

        for_in_t2(|i, j| {
            self.strain_fft[i][j].fft_assign(&self.strain[i][j], &mut self.fft_scratch, &self.ffts);
        });
    }
    pub fn step(&mut self) {
        for_in_t2(|i, j| {
            self.stress_fft[i][j].fft_assign(&self.stress[i][j], &mut self.fft_scratch, &self.ffts);
        });

        for_in_t4(|i, j, k, h| {
            for_in_field(|index| {
                let value = self.greens[i][j][k][h].get(index) * self.stress_fft[k][h].get(index);
                self.strain_fft[i][j].sub_assign(index, value);
            });
            #[cfg(feature = "d2")]
            let scale = (W * H) as f64;
            #[cfg(feature = "d3")]
            let scale = (W * H * D) as f64;
            self.strain_fft[i][j].set([0; DIM], (self.strain0[i][j] * scale).into());
        });

        for_in_t2(|i, j| {
            self.strain[i][j].ifft_assign(
                &self.strain_fft[i][j],
                &mut self.fft_scratch,
                &self.iffts,
            );
        });
        tensor_mul_assign(&mut self.stress, &self.strain, &self.c);
    }
    pub fn set_ddsdde(&self, ddsdde: &mut [[f64; NTENS]; NTENS]) {
        ddsdde.copy_from_slice(&self.ddsdde);
    }
    pub fn set_average_stress(&self, stress: &mut [f64; NTENS]) {
        #[cfg(feature = "d2")]
        for_in_field(|index| {
            stress[0] += self.stress[0][0].get(index).re;
            stress[1] += self.stress[1][1].get(index).re;
            stress[2] += self.stress[0][1].get(index).re;
        });
        #[cfg(feature = "d3")]
        for_in_field(|index| {
            stress[0] += self.stress[0][0].get(index).re;
            stress[1] += self.stress[1][1].get(index).re;
            stress[2] += self.stress[2][2].get(index).re;
            stress[3] += self.stress[0][1].get(index).re;
            stress[4] += self.stress[0][2].get(index).re;
            stress[5] += self.stress[1][2].get(index).re;
        });

        stress.iter_mut().for_each(|v| *v *= INV_N);
    }
    pub fn stress(&self) -> &Tensor2<Complex64> {
        &self.stress
    }
    pub fn strain(&self) -> &Tensor2<Complex64> {
        &self.strain
    }
    pub fn error(&self) -> f64 {
        convergence_error(&self.stress_fft, &self.freqs)
    }
}

fn init_c_from_image(
    lame1: f64,
    lame2: f64,
    mu1: f64,
    mu2: f64,
    img: &image::GrayImage,
) -> Tensor4<f64> {
    let mut c = Tensor4::<f64>::default();
    for_in_field(|index| {
        use crate::new::field::for_in_t2;

        let is_base = img.get_pixel(index[0] as u32, index[1] as u32).0[0] == 255;
        let lame = if is_base { lame1 } else { lame2 };
        let shear_modulus = if is_base { mu1 } else { mu2 };
        for_in_t2(|a, b| {
            c[a][a][b][b].add_assign(index, lame);
            c[a][b][a][b].add_assign(index, shear_modulus);
            c[a][b][b][a].add_assign(index, shear_modulus);
        });
    });
    c
}

// fn init_ordinary_c(lame: f64, mu: f64) -> Tensor4<f64> {
//     // let mut c1 = Tensor4::<f64>::default();
//     // for_in_t4(|i, j, k, h| {
//     //     for_in_field(|index| {
//     //         if i == j && k == h {
//     //             c1[i][j][k][h].set(index, lame);
//     //         }
//     //         if i == k && j == h {
//     //             c1[i][j][k][h].add_assign(index, mu);
//     //         }
//     //         if i == h && j == k {
//     //             c1[i][j][k][h].add_assign(index, mu);
//     //         }
//     //     });
//     // });
//     let mut c = Tensor4::<f64>::default();
//     for_in_t2(|a, b| {
//         for_in_field(|index| {
//             c[a][a][b][b].add_assign(index, lame);
//             c[a][b][a][b].add_assign(index, mu);
//             c[a][b][b][a].add_assign(index, mu);
//         });
//     });
//     // println!("eq? {}", c1 == c);
//     c
// }

fn init_ddsdde_from_image(
    lame1: f64,
    lame2: f64,
    mu1: f64,
    mu2: f64,
    img: &image::GrayImage,
) -> [[f64; NTENS]; NTENS] {
    let mut lame = 0.0;
    let mut mu = 0.0;
    for_in_field(|index| {
        let is_base = img.get_pixel(index[0] as u32, index[1] as u32).0[0] == 255;
        lame += if is_base { lame1 } else { lame2 };
        mu += if is_base { mu1 } else { mu2 };
    });

    lame *= INV_N;
    mu *= INV_N;

    let mut result: [[f64; NTENS]; NTENS] = Default::default();
    let a = 2.0 * mu + lame;
    result[0][0] = a;
    result[1][1] = a;
    result[2][2] = mu;
    result[0][1] = lame;
    result[1][0] = lame;
    result
}

fn convergence_error(stress_fft: &Tensor2<Complex64>, freq: &[Field<f64>; DIM]) -> f64 {
    let mut result = 0.0;
    for_in_field(|index| {
        let mut vector = [Complex64::ZERO; 2];
        for_in_t2(|i, j| {
            vector[j] += freq[i].get(index) * stress_fft[i][j].get(index);
        });
        result += vector[0].norm_sqr() + vector[1].norm_sqr();
    });
    result *= INV_N;
    result = result.sqrt();

    let denominator = {
        let mut vector = [Complex64::ZERO; 2];
        for_in_t2(|i, j| {
            vector[j] += stress_fft[i][j].get([0; DIM]);
        });
        (vector[0].norm_sqr() + vector[1].norm_sqr()).sqrt()
    };

    result / denominator
}
