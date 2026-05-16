use std::sync::Arc;

use rustfft::{
    Fft,
    num_complex::{Complex, Complex64},
};

pub const W: usize = 2;
pub const H: usize = 2;
#[cfg(feature = "d3")]
pub const D: usize = 2;

#[cfg(feature = "d2")]
pub const DIM: usize = 2;
#[cfg(feature = "d3")]
pub const DIM: usize = 3;

// fn field_to_complex(field: &Field<f64>) -> Field<Complex64> {
//     Field {
//         data: Box::new(field.data.map(|v| Complex64 { re: v, im: 0.0 })),
//     }
// }

pub fn field_to_real<T: Copy + Default>(f: &Field<Complex<T>>) -> Field<T> {
    let mut result: Field<T> = Default::default();
    for_in_field(|index| {
        result.set(index, f.get(index).re);
    });
    result
}

// Accessing: z, y, x
// index = z * W * H + y * W + x
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field<T> {
    #[cfg(feature = "d2")]
    pub data: Box<[T; W * H]>,
    #[cfg(feature = "d3")]
    pub data: Box<[T; W * H * D]>,
}
impl<T: Default + Copy> Default for Field<T> {
    fn default() -> Self {
        Self {
            #[cfg(feature = "d2")]
            data: Box::new([T::default(); W * H]),
            #[cfg(feature = "d3")]
            data: Box::new([T::default(); W * H * D]),
        }
    }
}

pub type Tensor4<T> = [[[[Field<T>; DIM]; DIM]; DIM]; DIM];
pub type Tensor2<T> = [[Field<T>; DIM]; DIM];

pub fn flat_index(index: [usize; DIM]) -> usize {
    #[cfg(feature = "d2")]
    let index = index[0] + index[1] * W;
    #[cfg(feature = "d3")]
    let index = index[0] + index[1] * W + index[2] * W * H;
    index
}

impl<T> Field<T> {
    pub fn set(&mut self, index: [usize; DIM], value: T) {
        self.data[flat_index(index)] = value;
    }
}
impl<T: Clone> Field<T> {
    pub fn get(&self, index: [usize; DIM]) -> T {
        self.data[flat_index(index)].clone()
    }
}

impl<T: Default + Copy> Field<T> {
    pub fn new_with(mut f: impl FnMut([usize; DIM]) -> T) -> Self {
        let mut result: Self = Default::default();
        for_in_field(|index| {
            result.set(index, f(index));
        });
        result
    }
}

impl<T: std::ops::AddAssign> Field<T> {
    pub fn add_assign(&mut self, index: [usize; DIM], value: T) {
        let index = flat_index(index);
        self.data[index] += value;
    }
}
impl<T: std::ops::SubAssign> Field<T> {
    pub fn sub_assign(&mut self, index: [usize; DIM], value: T) {
        let index = flat_index(index);
        self.data[index] -= value;
    }
}

pub fn for_in_field(mut f: impl FnMut([usize; DIM])) {
    #[cfg(feature = "d2")]
    for j in 0..H {
        for i in 0..W {
            f([i, j]);
        }
    }
    #[cfg(feature = "d3")]
    for k in 0..D {
        for j in 0..H {
            for i in 0..W {
                f([i, j, k])
            }
        }
    }
}

pub fn for_in_t4(mut f: impl FnMut(usize, usize, usize, usize)) {
    for i in 0..DIM {
        for j in 0..DIM {
            for k in 0..DIM {
                for h in 0..DIM {
                    f(i, j, k, h);
                }
            }
        }
    }
}

pub fn for_in_t2(mut f: impl FnMut(usize, usize)) {
    for i in 0..DIM {
        for j in 0..DIM {
            f(i, j);
        }
    }
}

pub fn tensor_mul_assign<
    R: Clone + std::ops::Mul<T, Output = T>,
    T: std::ops::AddAssign + Clone + num_traits::ConstZero,
>(
    lhs: &mut Tensor2<T>,
    rhs: &Tensor2<T>,
    m: &Tensor4<R>,
) {
    for_in_t2(|i, j| for_in_field(|index| lhs[i][j].set(index, T::ZERO)));
    for_in_t4(|i, j, k, h| {
        for_in_field(|index| {
            lhs[i][j].add_assign(index, m[i][j][k][h].get(index) * rhs[k][h].get(index));
        })
    });
}

impl<T: Clone> Field<T> {
    fn transpose_xy_to(&self, out: &mut Self) {
        #[cfg(feature = "d2")]
        for y in 0..H {
            for x in 0..W {
                let i1 = y * W + x;
                let i2 = x * H + y;
                out.data[i2] = self.data[i1].clone();
            }
        }
        #[cfg(feature = "d3")]
        for z in 0..D {
            let z_base = z * W * H;
            for y in 0..H {
                for x in 0..W {
                    let i1 = z_base + y * W + x;
                    let i2 = z_base + x * H + y;
                    out.data[i2] = self.data[i1].clone();
                }
            }
        }
    }
    fn transpose_yx_to(&self, out: &mut Self) {
        #[cfg(feature = "d2")]
        for y in 0..H {
            for x in 0..W {
                let i1 = y * W + x;
                let i2 = x * H + y;
                out.data[i1] = self.data[i2].clone();
            }
        }
        #[cfg(feature = "d3")]
        for z in 0..D {
            let z_base = z * W * H;
            for y in 0..H {
                for x in 0..W {
                    let i1 = z_base + y * W + x;
                    let i2 = z_base + x * H + y;
                    out.data[i1] = self.data[i2].clone();
                }
            }
        }
    }
    #[cfg(feature = "d3")]
    fn transpose_xz_to(&self, out: &mut Self) {
        for z in 0..D {
            for y in 0..H {
                for x in 0..W {
                    let i1 = z * W * H + y * W + x;
                    let i2 = x * D * H + y * D + z;
                    out.data[i2] = self.data[i1].clone();
                }
            }
        }
    }
    #[cfg(feature = "d3")]
    fn transpose_zx_to(&self, out: &mut Self) {
        for z in 0..D {
            for y in 0..H {
                for x in 0..W {
                    let i1 = z * W * H + y * W + x;
                    let i2 = x * D * H + y * D + z;
                    out.data[i1] = self.data[i2].clone();
                }
            }
        }
    }
}

impl Field<Complex64> {
    pub fn fft_assign(
        &mut self,
        source: &Self,
        scratch: &mut [Complex64],
        processors: &[Arc<dyn Fft<f64>>; DIM],
    ) {
        let step = W;
        #[cfg(feature = "d2")]
        let max = H * step;
        #[cfg(feature = "d3")]
        let max = D * H * step;

        let mut base = 0;
        while base < max {
            let next_base = base + step;
            processors[0].process_immutable_with_scratch(
                &source.data[base..next_base],
                &mut self.data[base..next_base],
                scratch,
            );
            base = next_base;
        }

        let mut temp = Field::default();
        self.transpose_xy_to(&mut temp);

        let step = H;
        #[cfg(feature = "d2")]
        let max = W * step;
        #[cfg(feature = "d3")]
        let max = D * W * step;

        let mut base = 0;
        while base < max {
            let next_base = base + step;
            processors[1].process_with_scratch(&mut temp.data[base..next_base], scratch);
            base = next_base;
        }

        temp.transpose_yx_to(self);

        #[cfg(feature = "d3")]
        {
            self.transpose_xz_to(&mut temp);

            let step = D;
            let max = W * H * step;

            let mut base = 0;
            while base < max {
                let next_base = base + step;
                processors[2].process_with_scratch(&mut temp.data[base..next_base], scratch);
                base = next_base;
            }

            temp.transpose_zx_to(self);
        }
    }

    pub fn ifft_assign(
        &mut self,
        source: &Self,
        scratch: &mut [Complex64],
        processors: &[Arc<dyn Fft<f64>>; DIM],
    ) {
        self.fft_assign(source, scratch, processors);

        #[cfg(feature = "d2")]
        const SCALE: f64 = 1.0 / (W * H) as f64;
        #[cfg(feature = "d3")]
        const SCALE: f64 = 1.0 / (W * H * D) as f64;

        self.data.iter_mut().for_each(|v| *v *= SCALE);
    }
}

pub fn get_freqs() -> [Field<f64>; DIM] {
    fn get_freq(i: usize, l: usize) -> f64 {
        let a = if i >= l / 2 {
            (i as isize - l as isize) as f64
        } else {
            i as f64
        };
        a / l as f64
    }

    let mut result: [Field<f64>; DIM] = Default::default();
    #[cfg(feature = "d3")]
    for z in 0..D {
        for y in 0..H {
            for x in 0..W {
                let index = z * H * W + y * W + x;
                result[0].data[index] = get_freq(x, W);
                result[1].data[index] = get_freq(y, H);
                result[2].data[index] = get_freq(z, D);
            }
        }
    }
    #[cfg(feature = "d2")]
    for y in 0..H {
        for x in 0..W {
            let index = y * W + x;
            result[0].data[index] = get_freq(x, W);
            result[1].data[index] = get_freq(y, H);
        }
    }
    result
}
