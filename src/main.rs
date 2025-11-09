use base64::{Engine as _, engine::general_purpose};
use clap::{Arg, Command};
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, warn};
use reqwest::blocking::Client;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

fn encode_image_to_base64(path: &Path) -> Result<(String, String), String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let mime_type = match ext.as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "bmp" => "image/bmp",
        "gif" => "image/gif",
        _ => return Err(format!("Unsupported image extension: .{}", ext)),
    };

    let b64_data = general_purpose::STANDARD.encode(&data);
    Ok((b64_data, mime_type.to_string()))
}

fn call_ollama_structured(
    client: &Client,
    api_url: &str,
    model: &str,
    images_b64: &[String],
    prompt: &str,
    schema_obj: Option<&Value>,
    options: Option<&Value>,
) -> Result<Value, String> {
    let messages = vec![json!({        "role": "user",
        "content": prompt,
        "images": images_b64,
    })];

    let mut payload = json!({
        "model": model,
        "messages": messages,
        "stream": false,
    });

    if let Some(schema) = schema_obj {
        payload["format"] = schema.clone();
    }
    if let Some(opts) = options {
        payload["options"] = opts.clone();
    }

    debug!(
        "Request payload: {}",
        serde_json::to_string_pretty(&payload).unwrap_or_default()
    );

    let resp = client
        .post(api_url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let status = resp.status();
    let text = resp.text().unwrap_or_default();

    if !status.is_success() {
        error!("Server said: {}", text);
        return Err(format!("HTTP error: {}", status));
    }

    serde_json::from_str(&text).map_err(|e| format!("Failed to parse JSON: {}", e))
}

fn main() {
    let matches = Command::new("Ollama Batch Image Captioning")
        .arg(
            Arg::new("dir")
                .long("dir")
                .required(true)
                .help("Directory of input images"),
        )
        .arg(
            Arg::new("api_url")
                .long("api_url")
                .required(true)
                .help("Ollama API URL"),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .required(true)
                .help("Model name in Ollama"),
        )
        .arg(
            Arg::new("schema")
                .long("schema")
                .help("JSON schema file path (optional)"),
        )
        .arg(
            Arg::new("prompt")
                .long("prompt")
                .default_value("What do you see in this image?")
                .help("Prompt to send to the model"),
        )
        .arg(
            Arg::new("output_dir")
                .long("output_dir")
                .help("Directory to save output JSON files"),
        )
        .arg(
            Arg::new("debug")
                .long("debug")
                .action(clap::ArgAction::SetTrue)
                .help("Enable verbose debug logging"),
        )
        .arg(
            Arg::new("options")
                .long("options")
                .help("JSON string of additional model options"),
        )
        .arg(
            Arg::new("pretty_json")
                .long("pretty-json")
                .action(clap::ArgAction::SetTrue)
                .help("Pretty format the JSON"),
        )
        .arg(
            Arg::new("batch_size")
                .long("batch-size")
                .default_value("1")
                .help("Number of images per batch"),
        )
        .arg(
            Arg::new("skip_existing")
                .long("skip-existing")
                .action(clap::ArgAction::SetTrue)
                .help("Skip any images which already have JSON for them."),
        )
        .arg(
            Arg::new("suffix")
                .long("suffix")
                .default_value("")
                .help("Suffix to append to JSON file names."),
        )
        .get_matches();

    let input_dir = matches.get_one::<String>("dir").unwrap();
    let api_url = matches.get_one::<String>("api_url").unwrap();
    let model = matches.get_one::<String>("model").unwrap();
    let prompt = matches.get_one::<String>("prompt").unwrap();
    let output_dir = matches
        .get_one::<String>("output_dir")
        .map(|s| s.as_str())
        .unwrap_or(input_dir);
    let pretty_json = matches.get_flag("pretty_json");
    let batch_size: usize = matches
        .get_one::<String>("batch_size")
        .unwrap()
        .parse()
        .unwrap_or(1);

    let skip_existing = matches.get_flag("skip_existing");
    let file_suffix = matches.get_one::<String>("suffix").unwrap();

    let schema_obj = matches.get_one::<String>("schema").map(|schema_path| {
        let schema_str = fs::read_to_string(schema_path).expect("Failed to read schema file");
        serde_json::from_str(&schema_str).expect("Invalid JSON schema")
    });

    let options = matches.get_one::<String>("options").map(|opts| {
        serde_json::from_str(opts).unwrap_or_else(|_| {
            warn!("Invalid JSON for --options; ignoring");
            json!({})
        })
    });

    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    let supported_ext: HashSet<&str> = ["jpg", "jpeg", "png", "bmp", "gif", "webp"]
        .iter()
        .cloned()
        .collect();
    let files: Vec<_> = fs::read_dir(input_dir)
        .expect("Failed to read input directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            let ext = path.extension()?.to_str()?.to_lowercase();

            if supported_ext.contains(ext.as_str()) {
                if skip_existing {
                    let json_path = Path::new(output_dir)
                        .join(format!("{}{}.json", path.file_stem().unwrap().to_string_lossy().to_string(), file_suffix));
                    if json_path.exists() {
                        return None;
                    } else {
                        return Some(path);
                    }
                } else {
                    Some(path)
                }
            } else {
                None
            }
        })
        .collect();

    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{wide_bar} {pos}/{len}")
            .expect("Error creating progress bar"),
    );

    let client = Client::new();

    for batch in files.chunks(batch_size) {
        let mut images_b64 = Vec::new();
        let mut batch_names = Vec::new();

        for path in batch {
            match encode_image_to_base64(path) {
                Ok((b64, _mime)) => {
                    images_b64.push(b64);
                    batch_names.push(path.file_stem().unwrap().to_string_lossy().to_string());
                }
                Err(e) => {
                    error!("Error encoding {}: {}", path.display(), e);
                    pb.inc(1);
                }
            }
        }

        if images_b64.is_empty() {
            continue;
        }

        match call_ollama_structured(
            &client,
            api_url,
            model,
            &images_b64,
            prompt,
            schema_obj.as_ref(),
            options.as_ref(),
        ) {
            Ok(resp) => {
                let contents = if let Some(messages) = resp.get("messages") {
                    messages
                        .as_array()
                        .map(|arr| {
                            arr.iter()
                                .map(|msg| msg.get("content").cloned().unwrap_or(json!("")))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                } else if let Some(message) = resp.get("message") {
                    vec![message.get("content").cloned().unwrap_or(json!(""))]
                } else {
                    vec![resp.clone()]
                };

                for (i, content) in contents.iter().enumerate() {
                    let output_data = if schema_obj.is_some() {
                        match serde_json::from_value::<Value>(content.clone()) {
                            Ok(val) => val,
                            Err(_) => {
                                warn!(
                                    "Response for {} is not valid JSON. Storing raw text.",
                                    batch_names[i]
                                );
                                json!(content)
                            }
                        }
                    } else {
                        content.clone()
                    };

                    let out_fname = Path::new(output_dir)
                        .join(format!("{}{}.json", batch_names[i], file_suffix));
                    let mut fo = File::create(&out_fname).expect("Failed to create output file");

                    let json_val = serde_json::from_str(output_data.as_str().unwrap())
                        .unwrap_or(output_data.clone());

                    let json_str = if pretty_json {
                        serde_json::to_string_pretty(&json_val).unwrap()
                    } else {
                        serde_json::to_string(&json_val).unwrap()
                    };
                    fo.write_all(json_str.as_bytes())
                        .expect("Failed to write output file");
                }
            }
            Err(e) => {
                error!("Error processing batch: {}", e);
            }
        }
        pb.inc(batch.len() as u64);
    }
    pb.finish_with_message("Done");
}
