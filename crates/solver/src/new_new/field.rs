use rustfft::num_complex::ComplexFloat;

pub const W: usize = 8;
pub const H: usize = 8;
#[cfg(feature = "d3")]
pub const D: usize = 2;

#[cfg(feature = "d2")]
pub const DIM: usize = 2;
#[cfg(feature = "d3")]
pub const DIM: usize = 3;

pub type Indices = [usize; DIM];

#[derive(Debug)]
pub struct Field<T> {
    #[cfg(feature = "d2")]
    pub data: Box<[[T; H]; W]>,
    #[cfg(feature = "d3")]
    pub data: Box<[[[T; D]; H]; W]>,
}

pub type Tensor4<T> = [[[[Field<T>; DIM]; DIM]; DIM]; DIM];

impl<T> Field<T> {
    pub fn set(&mut self, indices: Indices, value: T) {
        #[cfg(feature = "d2")]
        {
            self.data[indices[0]][indices[1]] = value;
        }
        #[cfg(feature = "d3")]
        {
            self.data[indices[0]][indices[1]][indices[2]] = value;
        }
    }
    pub fn get_mut(&mut self, indices: Indices) -> &mut T {
        #[cfg(feature = "d2")]
        {
            &mut self.data[indices[0]][indices[1]]
        }
        #[cfg(feature = "d3")]
        {
            &mut self.data[indices[0]][indices[1]]
        }
    }
}

impl<T: Copy> Field<T> {
    pub fn get(&self, indices: Indices) -> T {
        #[cfg(feature = "d2")]
        {
            self.data[indices[0]][indices[1]]
        }
        #[cfg(feature = "d3")]
        {
            self.data[indices[0]][indices[1]]
        }
    }
}

impl<T: Default + Copy> Default for Field<T> {
    fn default() -> Self {
        Self {
            data: Box::new([[T::default(); H]; W]),
        }
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

pub fn for_in_field(mut f: impl FnMut(Indices)) {
    #[cfg(feature = "d2")]
    for i in 0..W {
        for j in 0..H {
            f([i, j]);
        }
    }
    #[cfg(feature = "d3")]
    for i in 0..W {
        for j in 0..H {
            for k in 0..D {
                f([i, j, k]);
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

impl<T: ComplexFloat> Field<T>
where
    <T as ComplexFloat>::Real: Default,
{
    fn to_real(&self) -> Field<<T as ComplexFloat>::Real> {
        let mut result = Field::default();

        for_in_field(|indices| {
            result.set(indices, self.get(indices).re());
        });
        result
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
    for x in 0..W {
        for y in 0..H {
            for z in 0..D {
                result[0].data[x][y][z] = get_freq(x, W);
                result[1].data[x][y][z] = get_freq(y, H);
                result[2].data[x][y][z] = get_freq(z, D);
            }
        }
    }
    #[cfg(feature = "d2")]
    for x in 0..W {
        for y in 0..H {
            result[0].data[x][y] = get_freq(x, W);
            result[1].data[x][y] = get_freq(y, H);
        }
    }
    result
}

pub fn for_in_t2(mut f: impl FnMut(usize, usize)) {
    for i in 0..DIM {
        for j in 0..DIM {
            f(i, j);
        }
    }
}
