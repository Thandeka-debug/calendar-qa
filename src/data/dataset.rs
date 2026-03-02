use crate::data::loader::QADocument;
use burn::data::dataset::{Dataset, InMemDataset};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QASample {
    pub context: String,
    pub question: String,
    pub answer: String,
    pub answer_start: usize,
}

pub struct QADataset {
    dataset: InMemDataset<QASample>,
    tokenizer: Arc<Tokenizer>,
    max_seq_len: usize,
}

impl QADataset {
    pub fn new(documents: Vec<QADocument>, tokenizer: Tokenizer, max_seq_len: usize) -> Self {
        let mut samples = Vec::new();
        for doc in documents {
            let year = doc.filename.split(|c: char| !c.is_ascii_digit()).find(|s| !s.is_empty()).unwrap_or("2024").to_string();
            for (i, p) in doc.paragraphs.iter().enumerate() {
                if p.contains("GRADUATION") || p.contains("CEREMONY") {
                    samples.push(QASample {
                        context: p.clone(),
                        question: format!("When is graduation {}?", year),
                        answer: "December".to_string(),
                        answer_start: i,
                    });
                }
                if p.contains("Committee") && p.contains("meeting") {
                    let count = doc.entities.get("committees").map(|c| c.len()).unwrap_or(0);
                    samples.push(QASample {
                        context: p.clone(),
                        question: format!("How many committee meetings {}?", year),
                        answer: count.to_string(),
                        answer_start: i,
                    });
                }
            }
        }
        Self {
            dataset: InMemDataset::new(samples),
            tokenizer: Arc::new(tokenizer),
            max_seq_len,
        }
    }
}

impl Dataset<QASample> for QADataset {
    fn get(&self, index: usize) -> Option<QASample> {
        self.dataset.get(index)
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}