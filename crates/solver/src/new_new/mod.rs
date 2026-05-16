pub mod field;
#[cfg(feature = "d3")]
use field::D;
use field::{H, Tensor4, W, for_in_field, for_in_t2};

mod mechanics;

use mechanics::{get_lame, get_shear_modulus, init_greens};

#[cfg(feature = "d2")]
const NTENS: usize = 3;
#[cfg(feature = "d3")]
const NTENS: usize = 6;

#[cfg(feature = "d2")]
const INV_N: f64 = 1.0 / (W * H) as f64;
#[cfg(feature = "d3")]
const INV_N: f64 = 1.0 / (W * H * D) as f64;

pub struct Solver {}

// pub static TEST_FIELD: std::sync::OnceLock<&'static [u8]> = std::sync::OnceLock::new();

impl Solver {
    pub fn new(mat_img_path: &str) -> Self {
        let (e1, e2) = (1e6, 2e6);
        let (nu1, nu2) = (0.3, 0.3);
        let is_plain_stress = true;

        let lame1 = get_lame(e1, nu1, is_plain_stress);
        let lame2 = get_lame(e2, nu2, is_plain_stress);
        let mu1 = get_shear_modulus(e1, nu1);
        let mu2 = get_shear_modulus(e2, nu2);

        let freqs = field::get_freqs();

        let greens = init_greens(lame1, mu1, &freqs);

        // TEST_FIELD
        //     .set(bytemuck::cast_slice(Box::leak(
        //         greens[0][0][0][0].data.clone(),
        //     )))
        //     .unwrap();

        Self {}
    }
}

#[cfg(feature = "d2")]
fn init_c_from_image(
    lame1: f64,
    lame2: f64,
    mu1: f64,
    mu2: f64,
    img: &image::GrayImage,
) -> Tensor4<f64> {
    let mut c = Tensor4::<f64>::default();
    for_in_field(|index| {
        let is_base = img.get_pixel(index[0] as u32, index[1] as u32).0[0] == 255;
        let lame = if is_base { lame1 } else { lame2 };
        let shear_modulus = if is_base { mu1 } else { mu2 };
        for_in_t2(|a, b| {
            *c[a][a][b][b].get_mut(index) += lame;
            *c[a][b][a][b].get_mut(index) += shear_modulus;
            *c[a][b][b][a].get_mut(index) += shear_modulus;
        });
    });
    c
}

#[cfg(feature = "d2")]
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
