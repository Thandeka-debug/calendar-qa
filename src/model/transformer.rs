use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, Int},
};

#[derive(Config, Debug)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
}

impl TransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Transformer<B> {
        Transformer {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            positional_embedding: EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device),
            output_projection: LinearConfig::new(self.d_model, self.vocab_size).init(device),
            d_model: self.d_model,
            dropout: self.dropout,
        }
    }
}

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    token_embedding: Embedding<B>,
    positional_embedding: Embedding<B>,
    output_projection: Linear<B>,
    d_model: usize,
    dropout: f64,
}

impl<B: Backend> Transformer<B> {
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let token_embeds = self.token_embedding.forward(input_ids);
        let positions = Tensor::arange(0..token_embeds.dims()[1] as i64, &token_embeds.device())
            .reshape([1, token_embeds.dims()[1]])
            .repeat(&[token_embeds.dims()[0], 1]);
        let pos_embeds = self.positional_embedding.forward(positions);
        self.output_projection.forward((token_embeds + pos_embeds) / (self.d_model as f64).sqrt())
    }
}