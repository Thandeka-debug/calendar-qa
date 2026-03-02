mod data;
mod model;
mod training;
mod inference;

use anyhow::Result;
use burn::backend::Wgpu;
use clap::{Parser, Subcommand};
use data::{loader::DocumentLoader, dataset::QADataset};
use model::transformer::TransformerConfig;
use inference::predictor::QAPredictor;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::fs;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(short, long)]
        data_dir: PathBuf,
        #[arg(short, long, default_value_t = 10)]
        epochs: usize,
    },
    Ask {
        #[arg(short, long)]
        question: String,
        #[arg(short, long)]
        doc: PathBuf,
    },
    Interactive {
        #[arg(short, long)]
        doc: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Commands::Train { data_dir, epochs } => train_model(data_dir, epochs)?,
        Commands::Ask { question, doc } => answer_question(question, doc)?,
        Commands::Interactive { doc } => interactive_mode(doc)?,
    }
    Ok(())
}

fn train_model(data_dir: PathBuf, epochs: usize) -> Result<()> {
    println!("Loading documents from {:?}...", data_dir);
    let loader = DocumentLoader::new();
    let doc_paths: Vec<PathBuf> = std::fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().unwrap_or_default() == "docx")
        .collect();
    let documents = loader.load_documents(&doc_paths)?;
    println!("Loaded {} documents", documents.len());
    let tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
    let _dataset = QADataset::new(documents, tokenizer, 512);
    println!("Training simulation completed for {} epochs", epochs);
    Ok(())
}

fn answer_question(question: String, doc_path: PathBuf) -> Result<()> {
    println!("\nQuestion: {}", question);
    let loader = DocumentLoader::new();
    let metadata = fs::metadata(&doc_path)?;
    let mut predictor = None;

    if metadata.is_dir() {
        let doc_paths: Vec<PathBuf> = std::fs::read_dir(&doc_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().unwrap_or_default() == "docx")
            .collect();
        let documents = loader.load_documents(&doc_paths)?;
        let config = TransformerConfig {
            vocab_size: 30522,
            d_model: 128,
            max_seq_len: 512,
            dropout: 0.1,
        };
        let device = burn::backend::wgpu::WgpuDevice::default();
        let mut pred = QAPredictor::new(
            config.init::<Wgpu>(&device),
            Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            512,
        );
        for doc in &documents {
            pred.load_calendar_data(&doc.content, &doc.filename);
        }
        predictor = Some(pred);
    } else {
        let documents = loader.load_documents(&[doc_path])?;
        if documents.is_empty() {
            println!("No document found");
            return Ok(());
        }
        let config = TransformerConfig {
            vocab_size: 30522,
            d_model: 128,
            max_seq_len: 512,
            dropout: 0.1,
        };
        let device = burn::backend::wgpu::WgpuDevice::default();
        let mut pred = QAPredictor::new(
            config.init::<Wgpu>(&device),
            Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            512,
        );
        pred.load_calendar_data(&documents[0].content, &documents[0].filename);
        predictor = Some(pred);
    }

    if let Some(pred) = predictor {
        println!("Answer: {}", pred.answer_question(&question));
    } else {
        println!("Could not initialize.");
    }
    Ok(())
}

fn interactive_mode(doc_path: PathBuf) -> Result<()> {
    println!("Interactive Mode - Type 'quit' to exit\n");
    let loader = DocumentLoader::new();
    let metadata = fs::metadata(&doc_path)?;
    let mut predictor = None;

    if metadata.is_dir() {
        let doc_paths: Vec<PathBuf> = std::fs::read_dir(&doc_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().unwrap_or_default() == "docx")
            .collect();
        let documents = loader.load_documents(&doc_paths)?;
        println!("Loaded {} documents", documents.len());
        let config = TransformerConfig {
            vocab_size: 30522,
            d_model: 128,
            max_seq_len: 512,
            dropout: 0.1,
        };
        let device = burn::backend::wgpu::WgpuDevice::default();
        let mut pred = QAPredictor::new(
            config.init::<Wgpu>(&device),
            Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            512,
        );
        for doc in &documents {
            pred.load_calendar_data(&doc.content, &doc.filename);
        }
        predictor = Some(pred);
        println!("Ready!\n");
    }

    if let Some(pred) = predictor {
        loop {
            print!("> ");
            std::io::Write::flush(&mut std::io::stdout())?;
            let mut question = String::new();
            std::io::stdin().read_line(&mut question)?;
            let q = question.trim();
            if q == "quit" || q == "exit" {
                break;
            }
            if q.is_empty() {
                continue;
            }
            println!("\nAnswer: {}\n", pred.answer_question(q));
        }
    }
    Ok(())
}