use anyhow::Result;
use docx_rs::*;
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct QADocument {
    pub filename: String,
    pub content: String,
    pub paragraphs: Vec<String>,
    pub entities: HashMap<String, Vec<String>>,
}

pub struct DocumentLoader;

impl DocumentLoader {
    pub fn new() -> Self {
        Self
    }

    pub fn load_documents<P: AsRef<Path>>(&self, paths: &[P]) -> Result<Vec<QADocument>> {
        let mut documents = Vec::new();
        for path in paths {
            if let Some(ext) = path.as_ref().extension() {
                if ext == "docx" {
                    if let Ok(doc) = self.load_docx(path) {
                        documents.push(doc);
                    }
                }
            }
        }
        Ok(documents)
    }

    fn load_docx<P: AsRef<Path>>(&self, path: P) -> Result<QADocument> {
        let file_bytes = fs::read(&path)?;
        let doc = read_docx(&file_bytes)?;
        let mut content = String::new();
        let mut paragraphs = Vec::new();
        
        for document_child in doc.document.children {
            if let DocumentChild::Paragraph(paragraph) = document_child {
                for paragraph_child in paragraph.children {
                    if let ParagraphChild::Run(run) = paragraph_child {
                        for run_child in &run.children {
                            if let RunChild::Text(text) = run_child {
                                let text = text.text.trim().to_string();
                                if !text.is_empty() {
                                    content.push_str(&text);
                                    content.push(' ');
                                    paragraphs.push(text);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let filename = path.as_ref().file_name().unwrap_or_default().to_string_lossy().to_string();
        let entities = self.extract_entities(&content);
        Ok(QADocument { filename, content, paragraphs, entities })
    }

    fn extract_entities(&self, content: &str) -> HashMap<String, Vec<String>> {
        let mut entities = HashMap::new();
        let date_regex = Regex::new(r"\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{4}\b|\b\d{1,2}\b|\b\d{4}\b").unwrap();
        entities.insert("dates".to_string(), date_regex.find_iter(content).map(|m| m.as_str().to_string()).collect());
        
        let committee_regex = Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Committee|Board|Council|Forum)\b").unwrap();
        entities.insert("committees".to_string(), committee_regex.find_iter(content).map(|m| m.as_str().to_string()).collect());
        
        let event_regex = Regex::new(r"\b[A-Z][A-Z\s]+(?:DAY|GRADUATION|HOLIDAY|MEETING|WORKSHOP|CONFERENCE)\b").unwrap();
        entities.insert("events".to_string(), event_regex.find_iter(content).map(|m| m.as_str().to_string()).collect());
        
        entities
    }
}