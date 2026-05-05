use crate::new::field::{Field, Tensor4, for_in_field, for_in_t4};

pub fn get_lame(young: f64, poisson: f64, is_plain_stress: bool) -> f64 {
    young * poisson / ((1.0 + poisson) * (1.0 - (2.0 - is_plain_stress as u8 as f64) * poisson))
}

pub fn get_shear_modulus(young: f64, poisson: f64) -> f64 {
    young / (2.0 * (1.0 + poisson))
}

pub fn init_greens(lame: f64, shear_modulus: f64, freqs: &[Field<f64>; 2]) -> Tensor4<f64> {
    let freq2 = Field::<f64>::new_with(|index| {
        let x = freqs[0].get(index);
        let y = freqs[1].get(index);
        #[cfg(feature = "d3")]
        let z = freqs[2].get(index);

        let result = x * x + y * y;
        #[cfg(feature = "d3")]
        let result = result + z * z;

        result
    });
    let k1 = Field::<f64>::new_with(|index| 1.0 / (4.0 * shear_modulus * freq2.get(index)));
    let k2 = Field::<f64>::new_with(|index| {
        let freq2 = freq2.get(index);
        (lame + shear_modulus) / (shear_modulus * (lame + 2.0 * shear_modulus) * freq2 * freq2)
    });

    let mut result: Tensor4<f64> = Default::default();

    for_in_t4(|i, j, k, h| {
        let scalar = &mut result[i][j][k][h];

        for_in_field(|index| {
            let value = k2.get(index)
                * freqs[i].get(index)
                * freqs[j].get(index)
                * freqs[k].get(index)
                * freqs[h].get(index);
            scalar.set(index, -value)
        });
        if k == i {
            for_in_field(|index| {
                let value = k1.get(index) * freqs[h].get(index) * freqs[j].get(index);
                scalar.add_assign(index, value);
            });
        }
        if h == i {
            for_in_field(|index| {
                let value = k1.get(index) * freqs[k].get(index) * freqs[j].get(index);
                scalar.add_assign(index, value);
            });
        }
        if k == j {
            for_in_field(|index| {
                let value = k1.get(index) * freqs[h].get(index) * freqs[i].get(index);
                scalar.add_assign(index, value);
            });
        }
        if h == j {
            for_in_field(|index| {
                let value = k1.get(index) * freqs[k].get(index) * freqs[i].get(index);
                scalar.add_assign(index, value);
            });
        }
    });
    result
}
