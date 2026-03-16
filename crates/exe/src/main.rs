use rustfft::num_complex::{Complex, Complex64};
use rustfft::{Fft, FftPlanner};

use std::sync::Arc;

const W: usize = 32;
const H: usize = 32;

fn main() {
    let (e1, e2) = (1.0, 2.0);
    let nu = 0.3;
    let is_plain_stress = true;

    let lame1 = get_lame(e1, nu, is_plain_stress);
    let lame2 = get_lame(e2, nu, is_plain_stress);
    let mu1 = get_shear_modulus(e1, nu);
    let mu2 = get_shear_modulus(e2, nu);

    let freqs = get_freqs();

    let greens = init_greens(lame1, mu1, &freqs);

    let c = init_c_from_image(lame1, lame2, mu1, mu2);

    let strain_0 = [[1.0, 0.5], [0.5, 0.0]];
    let (stress, strain) = init_stress_strain(strain_0, &c);
    let (mut stress, mut strain) = (tensor22_to_complex(stress), tensor22_to_complex(strain));

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(W);
    let ifft = planner.plan_fft_inverse(W);
    let mut fft_scratch = [Complex64::default(); W];

    let mut stress_fft: Tensor22<Complex64> = Default::default();
    let mut strain_fft: Tensor22<Complex64> = Default::default();
    tensor22_fft_assign(&mut strain_fft, &strain, &mut fft_scratch, &fft);

    let mut step = || {
        tensor22_fft_assign(&mut stress_fft, &stress, &mut fft_scratch, &fft);

        println!("Error: {}", convergence_error(&stress_fft, &freqs));

        for_in_2222(|i, j, k, h| {
            for_in_field(|x, y| {
                let value = greens[i][j][k][h].get(x, y) * stress_fft[k][h].get(x, y);
                strain_fft[i][j].sub_assign(x, y, value);
            });
            strain_fft[i][j].set(0, 0, (strain_0[i][j] * (W * H) as f64).into());
        });

        tensor22_ifft_assign(&mut strain, &strain_fft, &mut fft_scratch, &ifft);
        tensor_mul_assign(&mut stress, &strain, &c);
    };

    for _ in 0..16 {
        step();
    }
    let stress = tensor22_to_real(&stress);
    for_in_22(|i, j| {
        stress[i][j].save_img(&format!("images/{i}{j}.png"));
    });
}

fn get_lame(young: f64, poisson: f64, is_plain_stress: bool) -> f64 {
    young * poisson / ((1.0 + poisson) * (1.0 - (2.0 - is_plain_stress as u8 as f64) * poisson))
}

fn get_shear_modulus(young: f64, poisson: f64) -> f64 {
    young / (2.0 * (1.0 + poisson))
}

struct Field2d<T> {
    data: Box<[[T; H]; W]>,
}

impl<T: Default + Copy> Default for Field2d<T> {
    fn default() -> Self {
        Self {
            data: Box::new([[T::default(); H]; W]),
        }
    }
}

type Tensor2222<T> = [[[[Field2d<T>; 2]; 2]; 2]; 2];
type Tensor22<T> = [[Field2d<T>; 2]; 2];

fn tensor22_to_complex<T: Default + Clone + num_traits::Zero + Copy>(
    t: Tensor22<T>,
) -> Tensor22<Complex<T>> {
    [
        [t[0][0].to_complex(), t[0][1].to_complex()],
        [t[1][0].to_complex(), t[1][1].to_complex()],
    ]
}

fn tensor22_to_real<T: Copy + Default>(t: &Tensor22<Complex<T>>) -> Tensor22<T> {
    [
        [t[0][0].real(), t[0][1].real()],
        [t[1][0].real(), t[1][1].real()],
    ]
}

fn tensor22_fft_assign(
    this: &mut Tensor22<Complex64>,
    source: &Tensor22<Complex64>,
    scratch: &mut [Complex64; W],
    fft: &Arc<dyn Fft<f64>>,
) {
    for_in_22(|i, j| {
        this[i][j].fft2_assign(&source[i][j], scratch, fft);
    });
}

fn tensor22_ifft_assign(
    this: &mut Tensor22<Complex64>,
    source: &Tensor22<Complex64>,
    scratch: &mut [Complex64; W],
    ifft: &Arc<dyn Fft<f64>>,
) {
    for_in_22(|i, j| {
        this[i][j].ifft2_assign(&source[i][j], scratch, ifft);
    });
}

impl<T> Field2d<T> {
    fn set(&mut self, i: usize, j: usize, value: T) {
        self.data[i][j] = value;
    }
}
impl<T: Clone> Field2d<T> {
    fn get(&self, i: usize, j: usize) -> T {
        self.data[i][j].clone()
    }
    fn transpose_inplace(&mut self) {
        for i in 0..W {
            for j in 0..i {
                let [s1, s2] = self.data.get_disjoint_mut([i, j]).unwrap();
                std::mem::swap(&mut s1[j], &mut s2[i]);
            }
        }
    }
}
impl Field2d<f64> {
    fn save_img(&self, path: &str) {
        let mut min = self.get(0, 0);
        let mut max = min;
        for_in_field(|i, j| {
            min = min.min(self.get(i, j));
            max = max.max(self.get(i, j));
        });
        let middle = (min + max) * 0.5;
        let inv_half = 1.0 / (middle - min);
        let diff = min * max < 0.0;
        let inv_range = 1.0 / (max - min);

        let mut image = image::Rgb32FImage::new(W as u32, H as u32);
        for_in_field(|x, y| {
            let value = self.get(x, y);
            let pixel = &mut image.get_pixel_mut(x as u32, y as u32).0;
            *pixel = [1.0, 1.0, 1.0];
            if diff {
                if value < middle {
                    let r = (middle - value) * inv_half;
                    pixel[0] -= r as f32 * 0.5;
                    pixel[1] -= r as f32;
                    pixel[2] -= r as f32;
                } else {
                    let b = (value - middle) * inv_half;
                    pixel[0] -= b as f32;
                    pixel[1] -= b as f32;
                    pixel[2] -= b as f32 * 0.5;
                }
            } else {
                let c = if min > 0.0 { value - min } else { max - value } * inv_range;
                pixel[0] -= c as f32;
                pixel[1] -= c as f32;
                pixel[2] -= c as f32;
            }
        });
        let image = image::DynamicImage::ImageRgb32F(image).into_rgb8();
        image.save(path).unwrap();
    }
}
impl Field2d<Complex64> {
    fn fft2_assign(
        &mut self,
        source: &Self,
        scratch: &mut [Complex64],
        processor: &Arc<dyn Fft<f64>>,
    ) {
        for i in 0..W {
            processor.process_immutable_with_scratch(&source.data[i], &mut self.data[i], scratch);
        }

        self.transpose_inplace();
        for i in 0..W {
            processor.process_with_scratch(&mut self.data[i], scratch);
        }
        self.transpose_inplace();
    }
    fn ifft2_assign(
        &mut self,
        source: &Self,
        scratch: &mut [Complex64],
        processor: &Arc<dyn Fft<f64>>,
    ) {
        for i in 0..W {
            processor.process_immutable_with_scratch(&source.data[i], &mut self.data[i], scratch);
        }

        self.transpose_inplace();
        for i in 0..W {
            processor.process_with_scratch(&mut self.data[i], scratch);
        }
        self.transpose_inplace();

        const SCALE: f64 = 1.0 / (W * H) as f64;
        self.data.iter_mut().flatten().for_each(|v| *v *= SCALE);
    }
}
impl<T: Default + Copy> Field2d<T> {
    fn new_with(mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut result = Self::default();
        for_in_field(|i, j| {
            let value = f(i, j);
            result.set(i, j, value);
        });
        result
    }
}

impl<T: Default + Clone + num_traits::Zero + Copy> Field2d<T> {
    fn to_complex(&self) -> Field2d<Complex<T>> {
        let mut result: Field2d<Complex<T>> = Default::default();
        for_in_field(|i, j| {
            result.set(
                i,
                j,
                Complex {
                    re: self.get(i, j),
                    im: T::zero(),
                },
            );
        });
        result
    }
}

impl<T: Copy + Default> Field2d<Complex<T>> {
    fn real(&self) -> Field2d<T> {
        let mut result: Field2d<T> = Default::default();
        for_in_field(|i, j| {
            result.set(i, j, self.get(i, j).re);
        });
        result
    }
}

impl<T: std::ops::AddAssign> Field2d<T> {
    fn add_assign(&mut self, i: usize, j: usize, value: T) {
        self.data[i][j] += value;
    }
}
impl<T: std::ops::SubAssign> Field2d<T> {
    fn sub_assign(&mut self, i: usize, j: usize, value: T) {
        self.data[i][j] -= value;
    }
}

fn for_in_field(mut f: impl FnMut(usize, usize)) {
    for i in 0..W {
        for j in 0..H {
            f(i, j);
        }
    }
}

fn for_in_2222(mut f: impl FnMut(usize, usize, usize, usize)) {
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for h in 0..2 {
                    f(i, j, k, h);
                }
            }
        }
    }
}

fn for_in_22(mut f: impl FnMut(usize, usize)) {
    for i in 0..2 {
        for j in 0..2 {
            f(i, j);
        }
    }
}

use std::fmt;
impl<T: fmt::Display> fmt::Debug for Field2d<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[\n")?;
        for i in 0..W {
            write!(f, "[")?;
            for j in 0..H - 1 {
                write!(f, "{}, ", self.data[i][j])?;
            }
            write!(f, "{}]\n", self.data[i][H - 1])?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

fn get_freqs() -> [Field2d<f64>; 2] {
    let mut result: [Field2d<f64>; 2] = Default::default();

    let inv_n = 1.0 / W as f64;

    let mut set_inner = |x: f64, i: usize| {
        let mut index = 0;
        for j in 0..=H / 2 - 1 {
            result[0].set(i, index, x);
            result[1].set(i, index, j as f64 * inv_n);
            index += 1;
        }
        for j in -(H as isize) / 2..=-1 {
            result[0].set(i, index, x);
            result[1].set(i, index, j as f64 * inv_n);
            index += 1;
        }
    };

    let mut index = 0;
    for i in 0..=W / 2 - 1 {
        set_inner(i as f64 * inv_n, index);
        index += 1;
    }
    for i in -(W as isize) / 2..=-1 {
        set_inner(i as f64 * inv_n, index);
        index += 1;
    }

    result
}

fn init_greens(lame: f64, shear_modulus: f64, freq: &[Field2d<f64>; 2]) -> Tensor2222<f64> {
    let freq2 = Field2d::<f64>::new_with(|i, j| {
        let x = freq[0].get(i, j);
        let y = freq[1].get(i, j);
        x * x + y * y
    });
    let k1 = Field2d::<f64>::new_with(|i, j| 1.0 / (4.0 * shear_modulus * freq2.get(i, j)));
    let k2 = Field2d::<f64>::new_with(|i, j| {
        let freq2 = freq2.get(i, j);
        (lame + shear_modulus) / (shear_modulus * (lame + 2.0 * shear_modulus) * freq2 * freq2)
    });

    let mut result: Tensor2222<f64> = Default::default();

    for_in_2222(|i, j, k, h| {
        let scalar = &mut result[i][j][k][h];
        for_in_field(|x, y| {
            let value = k2.get(x, y)
                * freq[i].get(x, y)
                * freq[j].get(x, y)
                * freq[k].get(x, y)
                * freq[h].get(x, y);
            scalar.set(x, y, -value)
        });
        if k == i {
            for_in_field(|x, y| {
                let value = k1.get(x, y) * freq[h].get(x, y) * freq[j].get(x, y);
                scalar.add_assign(x, y, value);
            });
        }
        if h == i {
            for_in_field(|x, y| {
                let value = k1.get(x, y) * freq[k].get(x, y) * freq[j].get(x, y);
                scalar.add_assign(x, y, value);
            });
        }
        if k == j {
            for_in_field(|x, y| {
                let value = k1.get(x, y) * freq[h].get(x, y) * freq[i].get(x, y);
                scalar.add_assign(x, y, value);
            });
        }
        if h == j {
            for_in_field(|x, y| {
                let value = k1.get(x, y) * freq[k].get(x, y) * freq[i].get(x, y);
                scalar.add_assign(x, y, value);
            });
        }
    });

    result
}

fn init_c_from_image(lame1: f64, lame2: f64, mu1: f64, mu2: f64) -> Tensor2222<f64> {
    use image::ImageReader;
    let img = ImageReader::open("images/mat.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma8();

    let mut c = Tensor2222::<f64>::default();
    for_in_field(|x, y| {
        let is_base = img.get_pixel(x as u32, y as u32).0[0] == 255;
        let lame = if is_base { lame1 } else { lame2 };
        let shear_modulus = if is_base { mu1 } else { mu2 };
        for_in_22(|a, b| {
            c[a][a][b][b].add_assign(x, y, lame);
            c[a][b][a][b].add_assign(x, y, shear_modulus);
            c[a][b][b][a].add_assign(x, y, shear_modulus);
        });
    });
    c
}

// fn stiffness_c(lame: f64, shear_modulus: f64) -> Tensor2222<f64> {
//     let mut c = Tensor2222::<f64>::default();
//     for_in_22(|a, b| {
//         for_in_field(|i, j| c[a][a][b][b].add_assign(i, j, lame));
//         for_in_field(|i, j| c[a][b][a][b].add_assign(i, j, shear_modulus));
//         for_in_field(|i, j| c[a][b][b][a].add_assign(i, j, shear_modulus));
//     });
//     c
// }

fn tensor_mul_assign<
    R: Clone + std::ops::Mul<T, Output = T>,
    T: std::ops::AddAssign + Clone + num_traits::ConstZero,
>(
    lhs: &mut Tensor22<T>,
    rhs: &Tensor22<T>,
    m: &Tensor2222<R>,
) {
    for_in_22(|i, j| for_in_field(|x, y| lhs[i][j].set(x, y, T::ZERO)));
    for_in_2222(|i, j, k, h| {
        for_in_field(|x, y| {
            lhs[i][j].add_assign(x, y, m[i][j][k][h].get(x, y) * rhs[k][h].get(x, y));
        })
    });
}

fn init_stress_strain(
    average_strain: [[f64; 2]; 2],
    c: &Tensor2222<f64>,
) -> (Tensor22<f64>, Tensor22<f64>) {
    let mut strain = Tensor22::<f64>::default();
    for_in_22(|i, j| for_in_field(|x, y| strain[i][j].set(x, y, average_strain[i][j])));
    let mut stress = Tensor22::<f64>::default();
    tensor_mul_assign(&mut stress, &strain, c);
    (stress, strain)
}

fn convergence_error(stress_fft: &Tensor22<Complex64>, freq: &[Field2d<f64>; 2]) -> f64 {
    let mut result = 0.0;
    for_in_field(|x, y| {
        let mut vector = [Complex64::ZERO; 2];
        for_in_22(|i, j| {
            vector[j] += freq[i].data[x][y] * stress_fft[i][j].data[x][y];
        });
        result += vector[0].norm_sqr() + vector[1].norm_sqr();
    });
    result /= (W * H) as f64;
    result = result.sqrt();

    let denominator = {
        let mut vector = [Complex64::ZERO; 2];
        for_in_22(|i, j| {
            vector[j] += freq[i].data[0][0] * stress_fft[i][j].data[0][0];
        });
        (vector[0].norm_sqr() + vector[1].norm_sqr()).sqrt()
    };
    println!("{}", denominator);

    // result / denominator
    result
}
