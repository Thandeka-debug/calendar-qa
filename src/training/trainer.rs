use crate::data::dataset::QASample;

pub struct TrainerConfig {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub seed: u64,
}

impl TrainerConfig {
    pub fn new(num_epochs: usize, batch_size: usize, learning_rate: f64) -> Self {
        Self {
            num_epochs,
            batch_size,
            learning_rate,
            seed: 42,
        }
    }
}

pub fn train_model(_data: &[QASample], _config: &TrainerConfig) {
    println!("Training placeholder");
}