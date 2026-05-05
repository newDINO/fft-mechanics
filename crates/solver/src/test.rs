struct Tensor<T> {
    shape: Vec<usize>,
    data: Vec<T>,
}

enum PermuteError {
    SizeMisMatch,
    AxesLenMisMatch,
    AxisOutOfBound,
}

impl<T: Clone> Tensor<T> {
    fn permute_to(&self, out: &mut Self, axes: &[usize]) -> Result<(), PermuteError> {
        if out.data.len() != self.data.len() {
            return Err(PermuteError::SizeMisMatch);
        }
        if axes.len() != self.shape.len() {
            return Err(PermuteError::AxesLenMisMatch);
        }
        Ok(())
    }
    fn permute_to_inner(
        &self,
        out: &mut Self,
        left_axes: &[usize],
        base0: usize,
        base1: usize,
    ) -> Result<(), PermuteError> {
        if let Some(axis) = left_axes.first() {
            let len = self.shape.get(*axis).ok_or(PermuteError::AxisOutOfBound)?;
            for i in 0..*len {
                let new_base0 = base0;
            }
        } else {
        }
        Ok(())
    }
}
