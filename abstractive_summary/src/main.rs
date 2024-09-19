use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::BufReader;
use std::env;
use std::path::PathBuf;
use anyhow::{Error as E, Result};
use tokenizers::{PaddingParams, TruncationParams, TruncationDirection, TruncationStrategy, Tokenizer};
use candle_transformers::models::quantized_t5 as t5;
use candle_core::{Device, Tensor};

const BATCH_SIZE: usize = 4;

#[derive(Serialize, Deserialize, Debug)]
struct InputData {
    text: String,
    id: i8,
}

fn make_input_data_from_file(filename: &str) -> Vec<InputData> {
    let file = File::open(filename).expect("Failed to open file");
    let reader = BufReader::new(file);
    let raw_input = serde_json::from_reader(reader).unwrap();

    raw_input
}

fn split_into_batch(texts: Vec<String>, batch_size: usize) -> Vec<Vec<String>> {
    let mut batches: Vec<Vec<String>> = Vec::new();
    let total_batch_number = texts.len() / batch_size;
    for i in 0..total_batch_number {
        let start = i * batch_size;
        let end = (i + 1) * batch_size;
        if end < texts.len() {
            batches.push(texts[start..end].to_vec());
        } else {
            batches.push(texts[start..].to_vec());
        }
    }

    batches
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: PathBuf,
}

impl T5ModelBuilder {
    pub fn load(model_path: String) -> Result<(Self, Tokenizer)> {
        let device = Device::Cpu;
        let config_filename = format!("{}/config.json", model_path);
        let tokenizer_filename = format!("{}/tokenizer.json", model_path);
        let weights_filename_string = format!("{}/model.gguf", model_path);
        let weights_filename = PathBuf::from(&weights_filename_string);
        let config = std::fs::read_to_string(config_filename)?;
        let config: t5::Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg).unwrap();
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))
    }

    pub fn build_model(&self) -> Result<t5::T5ForConditionalGeneration> {
        let device = Device::Cpu;
        let vb = t5::VarBuilder::from_gguf(&self.weights_filename, &device).unwrap();
        Ok(t5::T5ForConditionalGeneration::load(vb, &self.config).unwrap())
    }
}

fn summary_texts(texts: Vec<String>, batch_size: usize) -> Vec<String> {
    let (model_builder, mut tokenizer) = T5ModelBuilder::load("./src/summarizer_model".to_string()).unwrap();
    let mut model = model_builder.build_model().unwrap();
    let device = &model_builder.device;
    let tokenizer = tokenizer
    .with_padding(
        Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: Some(8),
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "<pad>".to_string(),
        }),
    )
    .with_truncation(Some(TruncationParams{
        direction: TruncationDirection::Right,
        max_length: 300,
        strategy: TruncationStrategy::LongestFirst,
        stride: 2,
    }))
    .map_err(E::msg).unwrap();

    let batches = split_into_batch(texts, batch_size);
    let mut tensor_batches: Vec<Tensor> = Vec::new();
    for i in 0..batches.len() {
        let batch = batches[i].clone();
        let encoded_batch = tokenizer
        .encode_batch_fast(batch, true)
        .map_err(E::msg).unwrap();
        
        let tensor_batch = Tensor::new(
            encoded_batch.into_iter().map(|encoding| {
                encoding.get_ids().iter().map(|&x| x).collect::<Vec<u32>>()
            }).collect::<Vec<Vec<u32>>>(),
            device,
        ).unwrap()
        .unsqueeze(0).unwrap();

        tensor_batches.push(tensor_batch);
    }

    let output_token_ids = vec![model_builder.config.decoder_start_token_id.unwrap_or(model_builder.config.pad_token_id) as u32; tensor_batches[0].shape().clone().into_dims()[2]];

    let mut summarized: Vec<String> = Vec::new();
    for (index, tensor_batch) in tensor_batches.iter().enumerate() {
        let decoder_token_ids = if index == 0 || !model_builder.config.use_cache {
            let batch_output_token_ids: Vec<Vec<Vec<u32>>> = vec![vec![output_token_ids.clone(); BATCH_SIZE]];
            Tensor::new(batch_output_token_ids, device).unwrap()
        } else {
            let last_token_vec:Vec<u32> = vec![*output_token_ids.last().unwrap()];
            let batch_output_token_ids: Vec<Vec<Vec<u32>>> = vec![vec![last_token_vec; BATCH_SIZE]];
            Tensor::new(batch_output_token_ids, device).unwrap()
        };
        println!("tensor_batch: {:?}", tensor_batch);
        println!("decoder_token_ids: {:?}", decoder_token_ids);
        let output_ids_vec = model
        .forward(&tensor_batch, &decoder_token_ids).unwrap()
        .to_vec2().unwrap();
        let output_ids_inner_unwrap = output_ids_vec.iter().map(|x| x.as_slice()).collect::<Vec<&[u32]>>();
        let output_ids = output_ids_inner_unwrap.as_slice();

        let output = tokenizer.decode_batch(output_ids, true).unwrap();
        for i in 0..output.len() {
            summarized.push(output[i].clone());
        }
    }

    summarized
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_file>", args[0]);
        return;
    }
    let filename = &args[1];
    let input = make_input_data_from_file(filename);
    
    let mut texts: Vec<String> = Vec::new();
    for i in 0..input.len() {
        texts.push(input[i].text.clone());
    }

    let summarized = summary_texts(texts, BATCH_SIZE);
    for i in 0..summarized.len() {
        println!("{}: {}", i, summarized[i]);
    }
}
