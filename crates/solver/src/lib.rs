use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

pub use rustfft::num_complex;

use std::sync::Arc;

// pub mod new;
// pub mod test;

pub const W: usize = 32;
pub const H: usize = 32;

pub struct Solver {
    fft: Arc<dyn Fft<f64>>,
    ifft: Arc<dyn Fft<f64>>,
    fft_scratch: [Complex64; W],
    freqs: [Field2d<f64>; 2],

    greens: Tensor2222<f64>,
    c: Tensor2222<f64>,
    ddsdde: [[f64; 3]; 3],

    strain0: [[f64; 2]; 2],

    stress: Tensor22<Complex64>,
    strain: Tensor22<Complex64>,
    stress_fft: Tensor22<Complex64>,
    strain_fft: Tensor22<Complex64>,
}

impl Solver {
    pub fn new(mat_img_path: &str) -> Self {
        let (e1, e2) = (1e6, 2e6);
        let nu = 0.3;
        let is_plain_stress = true;

        let lame1 = get_lame(e1, nu, is_plain_stress);
        let lame2 = get_lame(e2, nu, is_plain_stress);
        let mu1 = get_shear_modulus(e1, nu);
        let mu2 = get_shear_modulus(e2, nu);

        let freqs = get_freqs();

        let greens = init_greens(lame1, mu1, &freqs);

        use image::ImageReader;
        let img = ImageReader::open(mat_img_path)
            .unwrap()
            .decode()
            .unwrap()
            .into_luma8();
        let c = init_c_from_image(lame1, lame2, mu1, mu2, &img);
        let ddsdde = init_ddsdde_from_image(lame1, lame2, mu1, mu2, &img);

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(W);
        let ifft = planner.plan_fft_inverse(W);
        let fft_scratch = [Complex64::default(); W];

        Self {
            fft,
            ifft,
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
    pub fn init(&mut self, strain: &[f64; 3]) {
        self.strain0[0][0] = strain[0];
        self.strain0[1][1] = strain[1];
        self.strain0[0][1] = strain[2];
        self.strain0[1][0] = strain[2];
        for_in_22(|i, j| {
            for_in_field(|x, y| self.strain[i][j].set(x, y, self.strain0[i][j].into()))
        });
        tensor_mul_assign(&mut self.stress, &self.strain, &self.c);

        tensor22_fft_assign(
            &mut self.strain_fft,
            &self.strain,
            &mut self.fft_scratch,
            &self.fft,
        );
    }
    pub fn step(&mut self) {
        tensor22_fft_assign(
            &mut self.stress_fft,
            &self.stress,
            &mut self.fft_scratch,
            &self.fft,
        );

        for_in_2222(|i, j, k, h| {
            for_in_field(|x, y| {
                let value = self.greens[i][j][k][h].get(x, y) * self.stress_fft[k][h].get(x, y);
                self.strain_fft[i][j].sub_assign(x, y, value);
            });
            self.strain_fft[i][j].set(0, 0, (self.strain0[i][j] * (W * H) as f64).into());
        });

        tensor22_ifft_assign(
            &mut self.strain,
            &self.strain_fft,
            &mut self.fft_scratch,
            &self.ifft,
        );
        tensor_mul_assign(&mut self.stress, &self.strain, &self.c);
    }
    pub fn set_ddsdde(&self, ddsdde: &mut [[f64; 3]; 3]) {
        ddsdde.copy_from_slice(&self.ddsdde);
    }
    pub fn set_average_stress(&self, stress: &mut [f64; 3]) {
        const INV_N: f64 = 1.0 / (W * H) as f64;
        for_in_field(|x, y| {
            stress[0] += self.stress[0][0].get(x, y).re;
            stress[1] += self.stress[1][1].get(x, y).re;
            stress[2] += self.stress[0][1].get(x, y).re;
        });
        stress.iter_mut().for_each(|v| *v *= INV_N);
    }
    pub fn stress(&self) -> &Tensor22<Complex64> {
        &self.stress
    }
    pub fn error(&self) -> f64 {
        convergence_error(&self.stress_fft, &self.freqs)
    }
}

fn get_lame(young: f64, poisson: f64, is_plain_stress: bool) -> f64 {
    young * poisson / ((1.0 + poisson) * (1.0 - (2.0 - is_plain_stress as u8 as f64) * poisson))
}

fn get_shear_modulus(young: f64, poisson: f64) -> f64 {
    young / (2.0 * (1.0 + poisson))
}

pub struct Field2d<T> {
    pub data: Box<[[T; H]; W]>,
}

impl<T: Default + Copy> Default for Field2d<T> {
    fn default() -> Self {
        Self {
            data: Box::new([[T::default(); H]; W]),
        }
    }
}

pub fn field_to_real(f: &Field2d<Complex64>) -> Field2d<f64> {
    let mut result = Field2d::default();
    for_in_field(|x, y| {
        result.set(x, y, f.get(x, y).re);
    });
    result
}

pub type Tensor2222<T> = [[[[Field2d<T>; 2]; 2]; 2]; 2];
pub type Tensor22<T> = [[Field2d<T>; 2]; 2];

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
    pub fn save_img(&self, path: &str) {
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

fn init_c_from_image(
    lame1: f64,
    lame2: f64,
    mu1: f64,
    mu2: f64,
    img: &image::GrayImage,
) -> Tensor2222<f64> {
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

fn init_ddsdde_from_image(
    lame1: f64,
    lame2: f64,
    mu1: f64,
    mu2: f64,
    img: &image::GrayImage,
) -> [[f64; 3]; 3] {
    let mut lame = 0.0;
    let mut mu = 0.0;
    for_in_field(|x, y| {
        let is_base = img.get_pixel(x as u32, y as u32).0[0] == 255;
        lame += if is_base { lame1 } else { lame2 };
        mu += if is_base { mu1 } else { mu2 };
    });
    let n = (W * H) as f64;
    lame /= n;
    mu /= n;

    let mut result: [[f64; 3]; 3] = Default::default();
    let a = 2.0 * mu + lame;
    result[0][0] = a;
    result[1][1] = a;
    result[2][2] = mu;
    result[0][1] = lame;
    result[1][0] = lame;
    result
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
            vector[j] += stress_fft[i][j].data[0][0];
        });
        (vector[0].norm_sqr() + vector[1].norm_sqr()).sqrt()
    };

    result / denominator
}
